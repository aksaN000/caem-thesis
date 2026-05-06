"""
caem/verification/alias_overlap.py
====================================
v2 Fix 6 — alias-overlap signal for the verifier composite.

Computes the fraction of named entities in the answer that are
recognised as Wikidata aliases of entities mentioned in the retrieved
top passages. High alias_overlap means the answer's surface forms are
canonical alternates of entities the retrieved evidence actually talks
about — strong signal that the answer is grounded in the evidence
beyond what string match (p_ground_max) and NLI entailment (p_entail)
can detect.

Why a separate signal
---------------------
String matching (used by p_ground_*) credits "Bill Clinton" for matching
"Bill Clinton" but not for matching "William Jefferson Clinton", even
though they refer to the same Wikidata entity Q1124. NLI entailment
(p_entail) sometimes catches alias relationships when the passage spells
out both surface forms, but routinely misses them when the passage uses
only one form. Wikidata aliases are the canonical multi-surface mapping
for the named-entity axis.

Resolver interface
------------------
The signal is computed via an injected ``AliasResolver`` so the
production path (a 150 MB Wikidata alias dump file) and the test path
(a tiny in-memory dict) share the same scorer code. Both implement::

    resolve(entity: str) -> Set[str]

Returning the canonical Wikidata aliases for ``entity``. The empty set
indicates "not in Wikidata"; the scorer treats unresolved entities as
non-overlapping.

Public surface
--------------
* :class:`AliasResolver` — the protocol every resolver must satisfy.
* :class:`InMemoryAliasResolver` — tiny dict-backed resolver (tests).
* :class:`WikidataAliasResolver` — production resolver, lazily loads
  the local Wikidata alias dump on first :meth:`resolve` call.
* :func:`compute_alias_overlap` — the actual scoring function. Defaults
  to a regex-based entity extractor that doesn't need spaCy installed
  (Title-Case sequences); the harness can pass an explicit
  ``entities_in_answer`` / ``entities_in_passages`` to override when a
  better extractor is available.

The resolver is loaded once and threaded through ``UnifiedVerifier``
so the cost of opening the alias dump is paid exactly once per process.
When the resolver is ``None`` (the default until Phase 1a wires it),
:func:`compute_alias_overlap` returns a neutral ``0.5`` so the composite
treats the signal as missing rather than penalising every answer.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Resolver protocol + concrete implementations                                  #
# ---------------------------------------------------------------------------- #

class AliasResolver(Protocol):
    """Protocol every alias resolver must satisfy."""

    def resolve(self, entity: str) -> Set[str]:
        """Return the set of Wikidata aliases for ``entity``.

        The returned set should INCLUDE the canonical surface form
        (so ``"William Jefferson Clinton" in resolve("Bill Clinton")``
        is True). Lookup is case-insensitive on the input but the
        returned aliases preserve their canonical casing for display.
        Returns an empty set when ``entity`` is not in Wikidata.
        """
        ...


class InMemoryAliasResolver:
    """Dict-backed resolver. Useful for tests and small deployments.

    The provided table is treated as the authoritative bidirectional
    alias graph: any surface form keyed in the dict resolves to the
    union of all values reachable from it. So given::

        {
            "Bill Clinton": {"William Jefferson Clinton", "Clinton"},
            "William Jefferson Clinton": {"Bill Clinton"},
        }

    ``resolve("Bill Clinton")`` returns
    ``{"Bill Clinton", "William Jefferson Clinton", "Clinton"}``.
    """

    def __init__(self, table: Optional[Dict[str, Iterable[str]]] = None) -> None:
        self._table: Dict[str, Set[str]] = {}
        if table:
            for k, vs in table.items():
                self._table.setdefault(k.casefold(), set()).update(vs)
                self._table[k.casefold()].add(k)
                for v in vs:
                    self._table.setdefault(v.casefold(), set()).update(vs)
                    self._table[v.casefold()].add(k)
                    self._table[v.casefold()].add(v)

    def resolve(self, entity: str) -> Set[str]:
        return set(self._table.get(entity.casefold(), set()))


class WikidataAliasResolver:
    """Lazy-loading resolver backed by a local Wikidata alias JSON dump.

    Expected JSON layout::

        {
            "<canonical_surface>": ["<alias_1>", "<alias_2>", ...],
            ...
        }

    The 150 MB Wikidata alias dump is downloaded once at deployment
    setup time (see ``scripts/download_wikidata_aliases.py`` — wired
    when Phase 1a runs). The dump is loaded into memory on first
    :meth:`resolve` call; subsequent calls hit the in-memory dict.
    """

    def __init__(self, alias_path: Path) -> None:
        self.alias_path = Path(alias_path)
        self._table: Optional[Dict[str, Set[str]]] = None

    def _load(self) -> Dict[str, Set[str]]:
        if not self.alias_path.is_file():
            logger.warning(
                "WikidataAliasResolver: alias file not found at %s; "
                "all resolves will return empty sets.",
                self.alias_path,
            )
            return {}
        logger.info("WikidataAliasResolver: loading %s ...", self.alias_path)
        with self.alias_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        table: Dict[str, Set[str]] = {}
        for canon, aliases in raw.items():
            entry = {canon} | set(aliases)
            for surface in entry:
                table.setdefault(surface.casefold(), set()).update(entry)
        logger.info(
            "WikidataAliasResolver: loaded %d surface forms.", len(table),
        )
        return table

    def resolve(self, entity: str) -> Set[str]:
        if self._table is None:
            self._table = self._load()
        return set(self._table.get(entity.casefold(), set()))


# ---------------------------------------------------------------------------- #
# Default entity extractor (regex; no spaCy dependency)                         #
# ---------------------------------------------------------------------------- #

_TITLE_RUN_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z'\-]+(?:\s+(?:of|the|de|von|van|du|la|le)\s+)?)+"
    r"(?:\s+[A-Z][a-zA-Z'\-]+)*\b"
)


def _default_entity_extractor(text: str) -> List[str]:
    """Extract title-cased entity-like spans from text.

    Coarse fallback when no spaCy/stanza extractor is available. Catches
    the common case of multi-word proper nouns ("New York", "Bill
    Clinton", "University of Cambridge") without external dependencies.
    Returns a list (preserves duplicates so downstream callers can
    decide whether to dedupe).
    """
    if not text:
        return []
    out: List[str] = []
    for m in _TITLE_RUN_RE.finditer(text):
        span = m.group(0).strip()
        # Drop bare 1-letter "I" / "A" matches and stop-word leakages.
        if len(span) >= 2 and span.lower() not in {"the", "a", "an"}:
            out.append(span)
    return out


# ---------------------------------------------------------------------------- #
# Scoring function                                                              #
# ---------------------------------------------------------------------------- #

def compute_alias_overlap(
    answer: str,
    passages: Iterable[str],
    resolver: Optional[AliasResolver] = None,
    *,
    entities_in_answer: Optional[List[str]] = None,
    entities_in_passages: Optional[List[str]] = None,
) -> float:
    """Compute the alias_overlap signal for one (answer, passages) pair.

    Definition::

        Let A = entities mentioned in the answer (after entity extraction).
        Let P = entities mentioned across all retrieved passages.
        For each a in A, mark it 'covered' iff
            a ∈ P  OR  resolve(a) ∩ aliases(P) ≠ ∅
        where aliases(P) = ⋃_{p ∈ P} resolve(p) ∪ {p}.

        alias_overlap = |covered A| / |A|     (or 0.5 if A is empty)

    Returns a float in ``[0, 1]``. The neutral fallback (0.5) is used
    when no resolver is configured, when the answer carries no entities,
    or when the passages carry no entities — matching ``q_a_relevance``'s
    "missing-signal" convention so the composite isotonic curve treats
    these as non-informative rather than catastrophically wrong.
    """
    if resolver is None:
        return 0.5

    if entities_in_answer is None:
        entities_in_answer = _default_entity_extractor(answer)
    if entities_in_passages is None:
        entities_in_passages = []
        for p in passages:
            entities_in_passages.extend(_default_entity_extractor(p))

    if not entities_in_answer:
        return 0.5
    if not entities_in_passages:
        return 0.5

    # Build the union of all alias-equivalent surfaces of any passage entity.
    passage_aliases: Set[str] = set()
    for p_ent in entities_in_passages:
        passage_aliases.add(p_ent.casefold())
        for alias in resolver.resolve(p_ent):
            passage_aliases.add(alias.casefold())

    covered = 0
    for a_ent in entities_in_answer:
        a_lower = a_ent.casefold()
        if a_lower in passage_aliases:
            covered += 1
            continue
        # Try the answer-side aliases against the passage-side surface set.
        a_aliases = {alias.casefold() for alias in resolver.resolve(a_ent)}
        if a_aliases & passage_aliases:
            covered += 1

    return float(covered) / float(len(entities_in_answer))
