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
            mc_judge = load_minicheck_judge(
                model_name=config.minicheck_model,
                device=device,
                entail_threshold=config.minicheck_entail_threshold,
                contradict_threshold=config.minicheck_contradict_threshold,
            )
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
            mc_judge = None

        if mc_judge is not None:
            # ---- Path B: try to attach FrozenQwenJudge via AdaptiveNLIJudge ----
            # Precondition: Platt calibration JSON must exist (produced by
            # scripts/calibrate_qwen_judge.py at step_platt_calibrate, which
            # runs after Step 7.0 and before Step 7 main). If the JSON is
            # absent we return the bare MiniCheckJudge so Step 7.0 works
            # unmodified; Step 7 main gets the full AdaptiveNLIJudge once
            # the calibration is in place.
            import json
            import os as _os
            platt_path = "outputs/calibration/qwen_judge_platt.json"
            use_adaptive = _os.environ.get("CAEM_USE_ADAPTIVE_JUDGE", "1") == "1"
            if use_adaptive and _os.path.isfile(platt_path):
                try:
                    with open(platt_path) as _f:
                        _platt = json.load(_f)
                    logger.info(
                        "Platt calibration found at %s (ρ=%.3f, MAE-logit=%.3f); "
                        "wrapping verifier judge in AdaptiveNLIJudge (MiniCheck "
                        "for hypothesis ≤408 MC-tokens, FrozenQwenJudge for longer).",
                        platt_path,
                        _platt.get("diagnostics", {}).get("pearson_rho_logit", 0.0),
                        _platt.get("diagnostics", {}).get("mae_logit", 0.0),
                    )
                    from caem.model_loader import load_base_generator
                    from caem.verification.qwen_judge import FrozenQwenJudge
                    from caem.verification.adaptive_nli_judge import AdaptiveNLIJudge
                    import torch as _torch

                    # Load FROZEN base Qwen weights (not the evolving SIL-training
                    # Qwen). See caem/verification/qwen_judge.py for why.
                    qwen_model, qwen_tok = load_base_generator(
                        config.base_model_name,
                        device=device,
                        dtype=_torch.bfloat16,
                    )
                    qwen_model.eval()
                    qwen_judge = FrozenQwenJudge(
                        qwen_model, qwen_tok, device=device,
                        platt_a=float(_platt.get("platt_a", 1.0)),
                        platt_b=float(_platt.get("platt_b", 0.0)),
                    )
                    adaptive = AdaptiveNLIJudge(
                        minicheck_judge=mc_judge,
                        qwen_judge=qwen_judge,
                        mc_tokenizer=mc_judge.tokenizer,
                    )
                    return adaptive, None, None
                except Exception as exc:
                    logger.warning(
                        "AdaptiveNLIJudge wiring failed (%s); returning bare "
                        "MiniCheck. Long-hypothesis samples may incur 512-token "
                        "truncation at the MiniCheck encoder.", exc,
                    )
                    return mc_judge, None, None
            else:
                logger.info(
                    "AdaptiveNLIJudge inactive (platt calibration missing or "
                    "CAEM_USE_ADAPTIVE_JUDGE=0); using bare MiniCheck judge. "
                    "This is the shipped Phase 1c+ configuration: the long-"
                    "hypothesis Qwen judge was ablated at Phase 2 after the "
                    "Platt-fit Pearson floor (rho >= 0.70) was not cleared "
                    "(measured rho = 0.5843, slope 0.137; see Appendix B "
                    "judge-ablation diagnostic). Step 7 main therefore runs "
                    "with bare MiniCheck for every prediction; long "
                    "predictions (>408 tokens) truncate at MiniCheck's 512-"
                    "token encoder, registered as a Threats item.",
                )
                return mc_judge, None, None

    # backend == "roberta_nli", or MiniCheck failed with fallback permitted.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    logger.info("Loading RoBERTa-NLI judge (%s)", config.nli_model)
    tokenizer = AutoTokenizer.from_pretrained(config.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.nli_model
    ).to(device)
    model.eval()
    return None, model, tokenizer
