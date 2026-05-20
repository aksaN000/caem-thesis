"""caem/routing/ruc.py
==========================
Retrieval Utility Classifier (RUC) — runtime inference module.

The deployed RUC is a Logistic Regression on the pre-routing-legal feature
set (see :class:`LRRetrievalUtilityClassifier`). It fires at the Tier-2 /
Tier-3 dispatch point: after Tier 1 (memory) misses, after a cheap top-1
FAISS lookup against the existing CAEM passage index, the classifier
predicts ``p_rag`` and the router escalates to Tier 3 iff ``p_rag >= τ``.

Tier 1 (memory exact match) and Tier 2 (similar-but-not-exact match) remain
untouched.

Inference cost per call: ~50-70 ms total — dominated by the FAISS top-1
search on the 21M-passage Wikipedia index (one-shot, page-cached after
warmup). The classifier itself is ~1 ms: StandardScaler.transform + LR
predict_proba on the feature vector.

Loaded artefacts (all frozen post-training):
    caem/ruc/v2/feature_spec.json
    caem/ruc/v2/models/scaler.joblib
    caem/ruc/v2/models/lr_final.joblib
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engineered-feature extraction — MUST mirror scripts/build_ruc_training_set.py
# exactly. Any drift here breaks the training/inference contract.
# ---------------------------------------------------------------------------

_INTERROGATIVES = ("who", "what", "when", "where", "why", "how", "yesno")
_YN_PREFIXES = re.compile(
    r"^\s*(is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\b",
    re.IGNORECASE,
)
_NEGATION_RX = re.compile(
    r"\b(not|never|no|none|nobody|nothing|neither|nor|n['’]?t)\b", re.IGNORECASE
)
_TEMPORAL_RX = re.compile(
    r"\b(recently|currently|now|today|yesterday|tomorrow|last\s+(week|month|year)|"
    r"nowadays|present|modern)\b|\b(19|20)\d{2}\b",
    re.IGNORECASE,
)
_NUMERIC_RX = re.compile(
    r"\b\d+(?:[.,]\d+)?\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|"
    r"hundred|thousand|million|billion)\b",
    re.IGNORECASE,
)
_MYTH_RX = re.compile(
    r"is\s+it\s+(true|safe|real|legal)|do\s+(people|most|some|many)\s+(believe|think|say)|"
    r"common\s+(misconception|myth|belief)|what\s+happens\s+if\s+you|"
    r"can\s+(you|i)\s+(get|catch|die\s+from)|"
    r"is\s+it\s+possible\s+to\s+",
    re.IGNORECASE,
)


def _interrogative_one_hot(question: str) -> Dict[str, float]:
    q = (question or "").strip().lower()
    out = {f"interrog_{k}": 0.0 for k in _INTERROGATIVES}
    for w in ("who", "what", "when", "where", "why", "how"):
        if q.startswith(w):
            out[f"interrog_{w}"] = 1.0
            return out
    if _YN_PREFIXES.search(q):
        out["interrog_yesno"] = 1.0
    return out


def _question_text_features(question: str) -> Dict[str, float]:
    q = question or ""
    feats: Dict[str, float] = {
        "q_token_len": float(len(q.split())),
        "negation_present": 1.0 if _NEGATION_RX.search(q) else 0.0,
        "temporal_cue": 1.0 if _TEMPORAL_RX.search(q) else 0.0,
        "numerical_cue": 1.0 if _NUMERIC_RX.search(q) else 0.0,
        "myth_regex_hit": 1.0 if _MYTH_RX.search(q) else 0.0,
    }
    feats.update(_interrogative_one_hot(q))
    return feats


# ---------------------------------------------------------------------------
# RUCDecision return type
# ---------------------------------------------------------------------------

@dataclass
class RUCDecision:
    """Output of one RUC inference.

    Attributes
    ----------
    decision : str
        Either ``"RAG"`` (route to Tier 3) or ``"DIRECT"`` (route to Tier 2).
    p_rag : float
        Calibrated probability that RAG is the better path for this query.
    threshold : float
        The ``τ_RUC`` threshold used to binarise (chosen at training time).
    features : Dict[str, float]
        The engineered feature vector that produced this decision. Stored
        for thesis logging.
    flavour : str
        Which ablation flavour shipped (A1/A2/A3/A4).
    """
    decision: str
    p_rag: float
    threshold: float
    features: Dict[str, float] = field(default_factory=dict)
    flavour: str = ""


# ---------------------------------------------------------------------------
# Logistic Regression + StandardScaler on the pre-routing feature set.
#
# Architectural note: this RUC fires AFTER the cheap top-1 FAISS lookup (which
# the pipeline can serve from the same PassageStore used by Tier 3). The
# top-1 search costs ~30-50 ms once the FAISS index is page-cached, far less
# than a full Tier-3 generation pass. The classifier's purpose is to decide
# whether to commit to that full Tier-3 path.
# ---------------------------------------------------------------------------

_ADV_RX = re.compile(
    r"\b(is\s+it\s+(true|safe|real|legal|possible|okay)|"
    r"do\s+(most\s+|some\s+|many\s+)?people\s+(believe|think|say|claim)|"
    r"can\s+you\s+(actually|really|truly)|"
    r"what\s+happens\s+if\s+you|"
    r"are\s+you\s+supposed\s+to|"
    r"common\s+(misconception|myth|belief))",
    re.IGNORECASE,
)
_OPI_RX = re.compile(
    r"\b(should|would|might|could|prefer|opinion|believe|feel|think)\b",
    re.IGNORECASE,
)
_OPEN_RX = re.compile(
    r"\b(why|how\s+to|describe|explain|discuss|"
    r"what\s+is\s+the\s+(meaning|purpose))\b",
    re.IGNORECASE,
)
_YEAR_RX = re.compile(r"\b(19|20)\d{2}\b")
_NUM_RX = re.compile(r"\b\d+(?:\.\d+)?\b")
_Q_EXPECTS_YEAR_RX = re.compile(
    r"\b(what\s+year|when\s+(did|was|were|is))\b", re.IGNORECASE
)
_Q_EXPECTS_NUMBER_RX = re.compile(
    r"\b(how\s+(many|much))\b", re.IGNORECASE
)


def _linguistic_features(question: str) -> Dict[str, float]:
    q = question or ""
    return {
        "is_adversarial_phrasing": 1.0 if _ADV_RX.search(q) else 0.0,
        "is_opinion_seeking":      1.0 if _OPI_RX.search(q) else 0.0,
        "is_open_ended":           1.0 if _OPEN_RX.search(q) else 0.0,
    }


def _answer_type_features(question: str, top1_passage: str) -> Dict[str, float]:
    q = question or ""
    p = top1_passage or ""
    qy = 1.0 if _Q_EXPECTS_YEAR_RX.search(q) else 0.0
    py = 1.0 if _YEAR_RX.search(p) else 0.0
    qn = 1.0 if _Q_EXPECTS_NUMBER_RX.search(q) else 0.0
    pn = 1.0 if _NUM_RX.search(p) else 0.0
    return {
        "q_expects_year":   qy,
        "passage_has_year": py,
        "year_match":       qy * py,
        "q_expects_number": qn,
        "passage_has_number": pn,
        "number_match":     qn * pn,
    }


class LRRetrievalUtilityClassifier:
    """Deployed RUC: Logistic Regression on the pre-routing feature set.

    Constructor takes pre-loaded dependencies so the pipeline shares them with
    the rest of CAEM (the runtime cost is one StandardScaler.transform + one
    LR.predict_proba, plus whatever the caller spent producing the feature
    inputs).

    Required feature inputs at predict time:

    - ``question`` (str): raw query.
    - ``top1_passage_text`` (str, optional): top-1 retrieved passage text.
      Pass ``""`` to fall back to the training-time stub (top1 similarity 0,
      entity overlap 0, no answer-type match). Recommended: caller hands in
      the result of ``PassageStore.search(qe, k=1)[0][0]``.
    - ``top1_passage_sim`` (float, default 0.0): cosine similarity of the
      top-1 retrieved passage to the question.
    - ``top1_passage_entity_overlap`` (float, default 0.0): fraction of
      question entities that appear in the top-1 passage.
    - ``p_ik`` (float, default 0.5): direct-correctness probe value. Pass
      0.5 as the no-information prior if the probe is not configured.

    The 14 surface-form features (length, regex, interrogatives, entities,
    pageviews) are computed inline from the question text and (optional)
    spaCy NER.
    """

    def __init__(
        self,
        scaler,
        lr,
        feature_order: List[str],
        spec: Dict[str, Any],
        nlp=None,
        pageview_table: Optional[Dict[str, float]] = None,
    ) -> None:
        self.scaler = scaler
        self.lr = lr
        self.feature_order = list(feature_order)
        self.spec = spec
        self.flavour = spec.get("ship_flavour", "LR_v2")
        self.threshold = float(spec.get("tau_RUC_default", 0.5))
        self.nlp = nlp
        self.pageview_table = pageview_table or {}

    @classmethod
    def from_spec(
        cls,
        spec_path: Path,
        *,
        load_spacy: bool = True,
    ) -> "LRRetrievalUtilityClassifier":
        import joblib

        spec = json.loads(Path(spec_path).read_text())
        if spec.get("model_type") != "logistic_regression_scaled":
            raise ValueError(
                f"Spec at {spec_path} has model_type "
                f"{spec.get('model_type')!r}; LRRetrievalUtilityClassifier "
                f"expects 'logistic_regression_scaled'."
            )
        scaler = joblib.load(spec["model_paths"]["scaler"])
        lr = joblib.load(spec["model_paths"]["lr"])

        nlp = None
        if load_spacy:
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
            except OSError:
                logger.warning(
                    "spaCy en_core_web_sm not loaded; entity_count will be "
                    "0.0 at inference (matches training-time stub when spaCy "
                    "is missing)."
                )

        return cls(
            scaler=scaler,
            lr=lr,
            feature_order=spec["features"],
            spec=spec,
            nlp=nlp,
        )

    def _feature_vector(
        self,
        question: str,
        *,
        top1_passage_text: str = "",
        top1_passage_sim: float = 0.0,
        top1_passage_entity_overlap: float = 0.0,
        p_ik: float = 0.5,
        **extra_features: float,
    ) -> Dict[str, float]:
        """Build the feature dict for one query.

        Features computed inline from the question text (surface form, NER,
        pageviews, linguistic patterns, answer-type heuristics) are always
        produced. Numerical features that require external compute (retrieval
        statistics, cross-encoder rerank, pairwise NLI, P(IK), Wikidata
        binary) are passed as keyword arguments by the caller; missing values
        fall back to the training-time neutral default of 0.0 (or 0.5 for
        ``p_ik``).
        """
        feats = _question_text_features(question)
        if self.nlp is not None:
            doc = self.nlp(question or "")
            feats["entity_count"] = float(len(doc.ents))
        else:
            feats["entity_count"] = 0.0
        # Pageview lookup: max log_pageviews over the question's entities,
        # or 0.0 if no entity matched. Matches training-time construction.
        log_pv = 0.0
        if self.nlp is not None and self.pageview_table:
            doc = self.nlp(question or "")
            for ent in doc.ents:
                v = self.pageview_table.get(ent.text.strip().lower())
                if v is not None and v > log_pv:
                    log_pv = float(v)
        feats["log_pageviews_max"] = log_pv
        feats["p_ik"] = float(p_ik)
        feats["top1_passage_sim"] = float(top1_passage_sim)
        feats["top1_passage_entity_overlap"] = float(top1_passage_entity_overlap)
        feats.update(_linguistic_features(question))
        feats.update(_answer_type_features(question, top1_passage_text))
        # Extra features (v2 additions: top5_*, rerank_*, nli_pair_*,
        # entity_in_wikidata, etc.) passed by the caller.
        for k, v in extra_features.items():
            feats[k] = float(v)
        return feats

    def predict(
        self,
        question: str,
        *,
        top1_passage_text: str = "",
        top1_passage_sim: float = 0.0,
        top1_passage_entity_overlap: float = 0.0,
        p_ik: float = 0.5,
        source_benchmark: Optional[str] = None,
        **extra_features: float,
    ) -> RUCDecision:
        """Decide RAG vs DIRECT for one query.

        Required: ``question``. Optional features (``top1_passage_*``, ``p_ik``,
        and any v2 additions like ``top5_sim_max``, ``rerank_top1``,
        ``nli_pair_mean``, ``entity_in_wikidata``, etc.) are supplied by the
        caller. Any feature listed in ``self.feature_order`` but not provided
        at call time defaults to 0.0 (or 0.5 for ``p_ik``).
        """
        feats = self._feature_vector(
            question,
            top1_passage_text=top1_passage_text,
            top1_passage_sim=top1_passage_sim,
            top1_passage_entity_overlap=top1_passage_entity_overlap,
            p_ik=p_ik,
            **extra_features,
        )
        x = np.array(
            [float(feats.get(f, 0.0)) for f in self.feature_order],
            dtype=np.float32,
        ).reshape(1, -1)
        x_s = self.scaler.transform(x)
        p_rag = float(self.lr.predict_proba(x_s)[0, 1])
        decision = "RAG" if p_rag >= self.threshold else "DIRECT"
        return RUCDecision(
            decision=decision,
            p_rag=p_rag,
            threshold=self.threshold,
            features=feats,
            flavour=self.flavour,
        )


__all__ = [
    "LRRetrievalUtilityClassifier",
    "RUCDecision",
]
