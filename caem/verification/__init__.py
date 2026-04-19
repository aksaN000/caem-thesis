"""CAEM verification layer: UnifiedVerifier + pluggable judge backends."""

from typing import Any, Optional, Tuple

from caem.config import CAEMConfig
from caem.verification.minicheck import _MiniCheckJudge, load_minicheck_judge
from caem.verification.verifier import (
    UnifiedVerifier,
    UnifiedVerifierOutput,
    _NLIEnsemble,
)

__all__ = [
    "UnifiedVerifier",
    "UnifiedVerifierOutput",
    "load_verifier_judge",
    "_MiniCheckJudge",
    "_NLIEnsemble",
]


def load_verifier_judge(
    config: CAEMConfig,
    device: str,
    *,
    allow_fallback: bool = True,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Load the verifier-side judge selected by config.verifier_backend.

    Returns a 3-tuple ``(judge, legacy_model, legacy_tokenizer)`` for
    back-compat with the UnifiedVerifier constructor. Exactly one of
    ``judge`` (new path) or ``(legacy_model, legacy_tokenizer)`` (RoBERTa
    fallback) is non-None:

        backend = "minicheck"  -> judge = _MiniCheckJudge, legacy_*=None
        backend = "roberta_nli"-> judge=None, legacy_model/tokenizer set

    Parameters
    ----------
    config : CAEMConfig
        Reads ``verifier_backend``, ``nli_model``, ``minicheck_model``,
        ``minicheck_entail_threshold``, ``minicheck_contradict_threshold``.
    device : str
        ``"cuda"`` or ``"cpu"``.
    allow_fallback : bool, default True
        When True and MiniCheck fails to load, silently fall back to
        RoBERTa and emit a loud warning. Set False in smoke tests where
        the verifier-backend must match the declared config exactly.
    """
    import logging
    logger = logging.getLogger(__name__)

    backend = (config.verifier_backend or "").strip().lower()

    if backend == "minicheck":
        try:
            judge = load_minicheck_judge(
                model_name=config.minicheck_model,
                device=device,
                entail_threshold=config.minicheck_entail_threshold,
                contradict_threshold=config.minicheck_contradict_threshold,
            )
            return judge, None, None
        except Exception as exc:
            if not allow_fallback:
                raise
            logger.warning(
                "MiniCheck judge failed to load (%s); falling back to "
                "RoBERTa-MNLI. This is NOT the thesis-default backend. "
                "Fix the MiniCheck load before running main results.",
                exc,
            )
            # Fall through to the RoBERTa path below.

    # backend == "roberta_nli", or MiniCheck failed with fallback permitted.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    logger.info("Loading RoBERTa-NLI judge (%s)", config.nli_model)
    tokenizer = AutoTokenizer.from_pretrained(config.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.nli_model
    ).to(device)
    model.eval()
    return None, model, tokenizer
