# Structural audit: ch5 L355–624 + ch6 (read-only)

Files: thesis_report/chapters/chapter_5.tex, chapter_6.tex, core/abstract.tex, appendix/appendix_d.tex, figures/auto/{tab_sig_test,tab_sig_test_c2,tab_baseline_pooled,tab_ruc_ablation,tab_tier_baseline_cost,tab_pooled_trajectory,tab_per_bench_trajectory,tab_tier_trajectory}.tex
All \Cref targets in range resolve (no undefined labels). Ranked by likelihood an examiner notices.

## 1. Statistical Analysis promises evidence ch5 never shows; Appendix D is never referenced; licensing rule defined two different ways  [concern (a)]
- L363–371 (sec:sig-mcnemar, eq:mcnemar): gives the McNemar chi-square formula. No b, c, chi-square or p appears anywhere in ch5. tab_sig_test(.tex/_c2) columns: n, EM_CAEM, EM_base, ΔEM, CHM_CAEM, CHM_base, ΔCHM, "Sig (EM / CHM)" (*, -, (B)), 95% CI. There are no p, no Holm-p and no chi-square columns. eq:mcnemar has 0 refs.
- L376: "Alongside the McNemar p-value, every per-benchmark contrast reports…". The p-value is not reported.
- L381: "the raw p-value and the Holm-adjusted p-value are both tabulated so that the reader can see the correction's effect". This is false for ch5. Appendix D gives Holm sequences for only 2 of 10 EM families (tab:app-sig-holm-tqa, tab:app-sig-holm-fever, C5 only) and chi-square/raw p for 4 sample contrasts (tab:app-sig-tables). Even those 4 rows have no b/c cells: the caption says "b and c are not persisted" and "χ² is … implied by the reported raw p-value". **No CHM (Wilcoxon) p-value is reported anywhere.**
- `app:significance` / `app:sig-*` have **0 references in the whole thesis**, so ch5 never points to Appendix D.
- Wilcoxon signed-rank on CHM is the test behind 21/20 of the headline significance wins (L394, both captions L404/L414), but no subsection of sec:significance describes it. It is described only in App D L63.
- **Licensing rule conflict.** ch5 L386: "the bootstrap 95% CI on the exact-match delta must lie entirely above 0.02". App D L144/L149/L153: "absolute exact-match difference (or absolute CHM difference) must clear 0.02", which is a point-estimate floor applied on both axes. The tables follow App D. Under ch5's own rule:
  - CHM stars with CIs that do not lie beyond ±0.02 would fail, e.g. C5 FEVER semantic_entropy CHM CI [-0.024, -0.002] and zero_shot [-0.026, -0.003], plus most CHM stars.
  - C2 EM stars at a lower bound of exactly +0.020 would also fail (TriviaQA flare [+0.020,+0.130]; TruthfulQA rag [+0.020,+0.137]).
  - So the 9/3, 21/3, 14/3 and 20/4 tallies hold only under App D's rule.
- L386 sets up three verdicts (supported, partially supported, unsupported). The tables show only */−, App D's "dagger" (significance-only) column is never shown, and no contrast is ever reported as "partially supporting".
- L386 also says a contrast verdict "support[s] the corresponding hypothesis from sec:hypotheses". None of H1–H5 (ch1 L131–139) is a baseline contrast, so the rule maps onto nothing (see #5).
- Fix:
  - Pick one rule. App D's rule is the one implemented, so rewrite L386 to match it.
  - Add raw-p and Holm-p columns (EM and CHM) to tab_sig_test/_c2, or put a full 80-row p table in App D.
  - Add "full contingency counts, raw and Holm-adjusted p-values: \Cref{app:significance}" to L381 and to both captions.
  - Add a short Wilcoxon subsection.
  - Delete or relabel the "partially supporting" tier, or show a dagger column.
  - Move the formula and procedural text of L358–386 to ch3/App D (see #16).

## 2. "Headline interpretation" (L554–557) is stale and conditional, and interprets things not shown  [concern (b)]
- The text reads: "will be drawn from…", "The interpretation that the architecture intends to license, conditional on the licensing rule … returning supported…", "The disclaimer that the licensing rule may return only partially supported…". This is pre-results registration prose. It reaches no verdict, even though sec:summary-headline (L491) already said "supported / supported with parity qualifier / partially supported" 60 lines earlier.
- It does not duplicate L491. It contradicts L491's tense and certainty, and it adds a claim that nothing in ch5 tests: "hallucination reduction **at the deployed cost envelope** is achievable … **rather than through scaling**".
  - No scaling contrast exists (single 3B backbone).
  - The cost side is not shown: sec:comp-tier-baseline is empty (#7), and H5 says latency stays in [13.6,14.7] s with "cost reduction not measurable" (L510).
- ch6 L27 (sec:concl-headline) turns the same sentence into an assertion: "therefore deliver hallucination reduction at the deployed cost envelope without scaling, and … converts … per-query cost from a flat tax … into a quantity that decreases across cycles". This contradicts H5 (L510) and the flat Tier-3 share in tab_tier_trajectory (51.3% → 49.7%).
- The same staleness appears elsewhere:
  - ch6 L27: "The headline finding the thesis **intends to** license".
  - ch6 L37 (closing remark): "If the licensing rule returns supported… if partially… if unsupported…", which is still a three-way conditional.
  - ch6 L20, contribution 3: "the supported / partially-supported / unsupported verdict … is gated by the main-run decision-distribution trajectory".
  - ch5 L624 signpost: "the framing under which the main-run trajectory and the comparative panel **will be** adjudicated".
  - ch6 L2–5 comment says "Status: SKELETON".
- Interpretation actually missing from the section: tab_pooled_trajectory shows CAEM C0 (cold start, before any self-improvement) at EM 0.541 against C2 0.544 and B1 0.507. Nearly all of the +7.4% EM over B1 is therefore present before the self-improvement loop runs. The headline does not separate the architectural effect (routing, memory, verifier) from the self-improvement effect, and an examiner will ask.
- There is a second gap at C5 on TruthfulQA. 64.3% of queries go to Tier 2 (zero-shot) with Tier-2 EM 0.321, still far below B1's 0.433. Retrieval amplification alone (L491, L602) cannot explain the regression, and the abstention explanation (L394, App D L155, abstract) competes with it. Neither text reconciles the two.
- Fix: rewrite L557 as the real interpretation in past tense:
  - (i) the gains on the CHM axis versus the EM axis;
  - (ii) most of the EM gain is present at C0, so the self-improvement loop mainly lowers CHM (0.135 → 0.107/0.109);
  - (iii) the TruthfulQA mechanism, reconciling abstention with retrieval amplification;
  - drop "rather than through scaling" and "deployed cost envelope", or move them to future work.
  - Rewrite ch6 L27/L37/L20 and ch5 L624 in past tense with actual verdicts.

## 3. "StrategyQA clean transfer-panel CHM win, −17.5%" is not significant under the thesis's own test
- Claimed in abstract L22 ("clean transfer-panel win of −17.5 percent"), ch5 L491 and ch6 L27.
- tab_sig_test.tex, zero_shot/strategyqa: ΔCHM −0.021, Sig "− / −", CI [−0.017, +0.002], which includes zero. Against cot, SE and B1 it is also "−".
- Fix: say "a point reduction of −17.5% that does not clear the licensing rule", or cut it from the abstract and ch6.
- Related: L491 calls the pooled EM against B1 "**supported** … at both reference cycles". At C5 the gain is +0.011 (+2.1%) with no test on pooled values, and per-benchmark against zero_shot at C5 there is 1 EM win (TriviaQA), 1 defeat (TruthfulQA) and 3 nulls. "Supported" here is not a licensing-rule verdict. Fix: drop the word or attach a pooled paired bootstrap.

## 4. "Parity with the strongest **fine-tuned** and active-retrieval baselines" is wrong for B6
- Claimed in abstract L22, ch5 L491 (twice) and ch6 L27.
- tab_baseline_pooled: B6 vanilla FT EM 0.490 against CAEM C5 0.517, a +0.027 (+5.6%) lead. Only B7 FLARE (0.525) edges C5. The table's own footnote says "C5 beats 7/8".
- Fix: change to "parity with the active-retrieval baseline (B7)" in all four places.

## 5. Hypothesis verdicts (L493–515) do not follow the registered rules
- L496 says "with the verdict drawn from the licensing rule of sec:sig-licensing". No row uses Holm or CI. The verdicts come from Cohen's d, π_store bands and Tier-1 counts.
- H2: ch1 L133 registers falsification as "any cycle in which the precondition holds and pool purity decreases". The trace in L507 is 0.746 → 0.740 → 0.736 → 0.746 → 0.732, which decreases at C1, C2 and C5. The verdict "supported (band)" re-specifies the hypothesis after the fact. This is a high-risk viva question.
- H4 and H5 are "scope-limited". ch1 L142 registers only "supported, partially supported, or unsupported".
- H5: "fires on 7 of 9000 … supporting the growth prediction" (Tier-1 share 0.0/0.0/0.1/0.1/0.1/0.1%). This is overclaiming. The cost-reduction half of H5 is explicitly not measurable.
- H1: ch1's deciding statistic is per-cycle, per-benchmark balanced accuracy. The verdict uses one cycle-zero d, plus π_store, which measures precision rather than balanced accuracy, as the "downstream receipt".
- Fix:
  - Delete "drawn from the licensing rule".
  - Report H2 as "unsupported as registered (non-monotone); consistent with the bounded-band restatement", or tag it partially supported.
  - Map scope-limited to "partially supported / not testable at this horizon" with a footnote.
  - Report per-cycle balanced accuracy for H1, or say why it is unavailable.

## 6. ch6 "Semantic-entropy hallucination metric" (L82–85, sec:future-se-chm) is stale: B8 CHM is reported, B6 CHM is the one missing
- L85: B8 "is paired with a deferred composite-hallucination-metric reading". But tab_baseline_pooled has B8 CHM 0.120 and both sig tables carry per-benchmark semantic_entropy CHM with stars and (B)s. L456 says the corrected B8 run supplies every B8 cell.
- The actually missing cells are **vanilla_ft CHM = nan / n/a** in tab_sig_test and _c2 (L491: "the five vanilla fine-tuning CHM cells are not available").
- Contradiction: tab_baseline_pooled nevertheless prints **B6 CHM 0.127** with deltas (−15.9% / −14.3%). Either the pooled cell lacks support or the per-benchmark nan is wrong.
- L85 also cites "the abstention-rate trajectory of sec:comp-semantic-entropy". That section (L456) reports no abstention rates.
- appendix_a.tex L331 and appendix_e.tex L170 also point to this stale item.
- Fix:
  - Retarget the item to the B6 CHM rescore.
  - Explain or remove B6 pooled CHM 0.127.
  - Fix the App A/E references.

## 7. "Tier-by-baseline cost relationship" (L480–483) reports nothing, yet is cited as evidence
- The prose sets up a comparison and then says "A consolidated cost-versus-accuracy comparison panel is deferred to follow-up work". It contains no number.
- figures/auto/tab_tier_baseline_cost.tex is a PLACEHOLDER stub ("Table pending main run") and is \input nowhere, so it is an orphan.
- It is cited as evidence in:
  - tab:evidence-mapping L537 ("Deployed cost-per-query decay across cycles → sec:tier-cost, sec:comp-tier-baseline");
  - ch5 L246 ("whose realised receipt is reported in sec:tier-cost and sec:comp-tier-baseline");
  - ch5 L600 ("the routing-signal axis is the architectural difference that the empirical contrast in sec:comp-tier-baseline … [is] registered to surface"). That section contains no such contrast.
- No ch6 future-work item covers the deferred cost panel.
- Fix:
  - Cut the subsection, or shrink it to one sentence inside sec:tier-cost.
  - Remove it from the L537 mapping row and L246.
  - Change L600 to point only to sec:ruc-ablation.
  - Add a ch6 item, or merge it into sec:future-latency.
  - Delete the stub file.

## 8. "Optional reference points" (L475–478) reports nothing and claims rows that do not exist
- "Both reference points are reported as separate rows in the comparative table". tab_baseline_pooled has no Self-RAG or STaR rows.
- STaR "run as an optional ceiling reference conditional on remaining compute budget": no result and no statement that it was not run.
- sec:comp-references has 0 refs.
- Fix: cut to a single sentence in Related Work/ch3 ("Self-RAG and STaR are discussed but not replicated because…"), or say plainly that STaR was not run.

## 9. Bootstrap and pairing descriptions contradict Appendix D, and the tables show CIs that exclude their own point estimate
- On the CHM bootstrap, ch5 L376 says it is "approximate: it resamples only the baseline's per-subtype rates, with CAEM's composite hallucination value held fixed, so it can exclude the point difference". App D L63 says "McNemar, Wilcoxon and bootstrap … all read off a single per-contrast paired-difference vector", and L91 says the CHM bootstrap substitutes the paired per-sample difference. These two descriptions cannot both be true.
- Examples of point estimates outside their CIs in the tables:
  - C5 fiveshot_cot/FEVER ΔCHM −0.015, CI [−0.010, +0.012];
  - C5 cot/CSQA −0.004, CI [−0.002, +0.016];
  - C5 fiveshot_cot/CSQA +0.007, CI [+0.008, +0.025];
  - C2 cot/StrategyQA −0.001, CI [+0.002, +0.022].
- These CHM CIs therefore cannot carry the effect-size half of the licensing rule.
- On manifest mismatches, ch5 L361 says they "halt the comparison". App D L18/L28 says they are "skipped with a warning", with a fallback to the raw identifier intersection.
- L376: "a confidence interval whose lower bound straddles zero" is garbled (a bound cannot straddle).
- Fix:
  - Make App D and ch5 agree on what was actually run.
  - If the CHM CI is unpaired, say so in the table captions and do not use it for licensing.
  - Align the halt-versus-skip wording.

## 10. Every defeat is said to be against "high-EM RAG-free baselines (zero-shot, CoT, FLARE, SE)", but FLARE retrieves
- L394 (and App D L155) list FLARE among "RAG-free" baselines. B7 is active retrieval (L451).
- "Every significant defeat … confined to TruthfulQA" is true at the Holm level. But vanilla_ft has EM CIs entirely below zero on FEVER (C5 [−0.123,−0.010]) and CSQA (C5 [−0.120,−0.010]; C2 [−0.117,−0.007]). App D tab:app-sig-holm-fever shows raw p = 0.029 against Holm 0.234. One clause acknowledging this would defuse the question.
- Fix: rephrase to "high-EM baselines (B1, B2, B7, B8)" and add the B6 caveat.

## 11. RUC ablation: acceptance gates defined two different ways
- L463 says the gates are "recovery of the TruthfulQA exact-match **and composite-hallucination** cells toward the B1 floor without degrading the training-panel pooled reading by more than 2 pp on either axis".
- L473 says gate 1 is a pooled EM lift with no per-benchmark EM regression above 2 pp, and gate 2 is TruthfulQA EM reaching B1 0.433.
- The TruthfulQA CHM half of L463's gate failed (0.116 → 0.119, +2.9%, moving away from B1's 0.051) and is never adjudicated.
- L491 says the ablation is "preserving the training-panel pooled reading", but CSQA improves by +14 pp.
- L469 has an unbalanced parenthesis ("…routing distribution)").
- The 0.310 vs 0.433 numbers are consistent everywhere (L473, L491, L602, ch6 L75, App D L155).
- Fix: state the gates once, add the TruthfulQA-CHM verdict, and change "preserving" to "without degrading".

## 12. ch6 "Restated contributions with empirical receipts" (L17–22)
- "The thesis registered **five contributions** in ch:introduction". ch1 has no contributions list. It registers **seven specific objectives** (ch1 L104–124) and five hypotheses (ch1 L142, L418). The five ch6 items (composite, gate, decision tree, theory, artefact bundle) do not map onto the seven objectives: routing/RUC, retroactive review, SIL plus retention guard, and baselines/ablations are not restated.
- Receipt quality varies:
  - C1: concrete (d = +0.635, +0.173; sec:adj-gate exists).
  - C2: no number ("the calibration-fold storage precision … [is] reported alongside…"). It should cite 0.796 at n = 289 and the [0.732, 0.746] band.
  - C3: no number and stale ("verdict … is gated by the main-run decision-distribution trajectory"). sec:decision-distribution (ch5 L141) has the numbers: store share 42 → 59%, abstain 21 → 10%.
  - C4: directional only.
  - C5: points to app:reproducibility (exists).
- The sixth property (L22) cites a correct Tier-2 share (48.7 → 55.2 → 50.2). But "an increasing share of previously retrieval-dependent queries can be served without paying the retrieval cost" is not what the data shows: Tier-3 share goes 51.3 → 44.7 → 55.1 → 49.7, non-monotone and essentially flat.
- Fix: re-key the section to ch1's seven objectives (or add a contributions list to ch1), put a number on every receipt, and soften the redistribution claim.

## 13. Future work against limitations (ch6 L39–95 vs ch5 L559–619)
**Limitations or threats that say "registered as future work" but have no ch6 item:**
- L565 single seed: "cross-seed replication … registered as a future-work item". No multi-seed item.
- L565 cross-backbone "with a specific candidate backbone identified in ch:conclusion". ch6 L65 is a *scale-up* (7B/13B, same family), not a comparable-scale different generator, and names no candidate.
- L590 RUC misrouting: "evaluation under a held-out routing-axis benchmark is registered as future work in ch:conclusion". Not in ch6.
- L595 sensitivity to φ_safe and η_u/η_g: "registered as future-work". Not in ch6.
- L609 deferred-pool steady-state survivability "registered as a future-work measurement". Not in ch6; sec:future-equilibrium-horizon does not mention it.
- L613 flexible-match re-scoring (format learning) "is registered as future work". Not in ch6.
- L509 H4 sharp gap-to-asymptote test "registered as future work". Not explicit in ch6.
- L483 consolidated cost-versus-accuracy panel "deferred to follow-up work". Not in ch6.
- ch5 L246 corpus-floor proxy "registered as a deferred proxy in ch:conclusion". Not in ch6.
- No future item for: L585 storage-tail n, L609 temperature saturation (T = 20.086), or L614–618 literature limitations (verifier over-optimisation, systematic verifier errors, calibration-fold size, low chance floor).

**ch6 items that follow from no stated limitation:**
- sec:future-lora: justified "if the per-cycle training loss is observed to plateau". No such observation is reported. Tie it to L618 (adapter capacity) or cut it.
- sec:future-refutation: the unreachable contradiction subtype is stated at ch5 L80 but is not in Limitations or Threats. Add one line to Limitations.
- sec:future-se-chm: stale (#6).
- sec:future-deployment: refers to an "accompanying production runbook" that is not in the thesis or its appendices. Cite or cut.

**Pointer hygiene:** ch5 uses a generic \Cref{ch:conclusion} 38 times. Of the 11 sec:future-* labels, only sec:future-backbone (1 ref, from ch6) and sec:future-se-chm (2, App A/E) are referenced. Replace the generic pointers with the specific \Cref{sec:future-…}; this makes the gaps above visible.

## 14. Redundancy of headline statements (numbers agree except where flagged)
| Statement | Locations |
|---|---|
| Training panel C2 +20.3% EM / −29.4% CHM vs B1 | abstract L22; ch5 L138; L491; ch6 L27 |
| C5 +13.7% / −20.3% (training) | ch5 L138 (CHM only), L491; ch6 L27 |
| Full panel C2 +7.4% / −11.0%; C5 +2.1% / −9.3% | abstract L22 (C2 only); L491; ch6 L27 (C2 only); tab_pooled_trajectory footer; tab_baseline_pooled |
| StrategyQA −17.5% "clean win" | abstract; L491; ch6 L27 (not significant, #3) |
| CHM beats all baselines at both cycles / EM parity | abstract; L491; ch6 L27 ("fine-tuned" wrong, #4) |
| 9/3 · 21/3 · 14/3 · 20/4 tallies | L394; L414 caption; L491; abstract (EM only); App D L155 (counts verified against tables) |
| TruthfulQA retrieval-amplification mechanism | L138, L151, L394, L467, L471, L491, L590, L602; ch6 L27 |
| RUC 0.210 → 0.310 vs 0.433; 99 → 40% Tier-3 | L473, L491, L602; ch6 L27 ("ten pp"), L75; App D L155 |
| "Deployed cost envelope / without scaling" | L557; ch6 L27 (unsupported, #2) |

Training-panel EM peak inconsistency: L138 says the training-panel peak is 0.584 at **C1**, while L491, ch6 and the abstract present C2's +20.3% as the "best cycle" figure. The best cycle is defined on the five-benchmark pool, which is fine, but say so.

63/9000 override: consistent (tab_safety_override). 7/9000 Tier-1: consistent with 0.1% × 1500 × 4 cycles.

Fix:
- Keep the full statement once (sec:summary-headline, shortened to about half).
- ch6 L27 should restate in 3–4 sentences and point back.
- L602 "Retrieval-amplification" paragraph: drop the repeated RUC numbers and point to L467–473.
- ch6 L75 repeats L473 almost verbatim; cut it to the arm-isolation design.

## 15. Limitations duplicated across Threats, Limitations and ch6
- Single seed/backbone: L565 and L609.
- Benchmark coverage/long-form: L570, L609, ch6 L60.
- Long-hypothesis judge: L570, L609, ch6 L32, ch6 L45.
- Horizon: L580, L609 ("longer trajectories are registered as future-work"), ch6 L95.
- Cal–eval gap: L575, L609 (exchangeability), ch6 L32 (open question 1).
- Deployment/live traffic: L609, ch6 L32 (open question 4), ch6 L70.
- Fix: Threats keeps the validity threats (seed, coverage, gap, horizon, n, RUC). Limitations keeps only what Threats lacks (memory scale, T saturation, deferred pool, the literature list). ch6 Open questions keeps only genuinely open theory questions (refit dynamics, fixed composite) and points back for the rest.

## 16. Misplaced content
- L358–386 (pairing, McNemar formula, bootstrap protocol, Holm procedure, licensing rule) is methodology. It belongs in ch3 or App D (which already duplicates it, L11–153). Leave a 1-paragraph results-facing summary in ch5 with p-value pointers.
- L418–456, per-baseline B1–B8 subsubsections: design rationale plus "expected contrast" predictions (L426, L431, L436, L446, L451, L456). All 0 refs.
  - Only B2 (L426) and B8 (L456) are adjudicated.
  - B6's prediction ("CAEM's retention ratio is strictly higher", L446) is never measured or reported.
  - B3, B4, B5 and B7 predictions are never checked against tab_sig_test.
  - Fix: move the descriptions to ch3/ch4 (baseline specification) and either add a one-line realised-vs-expected verdict per baseline or cut the expectations.
- L456: the B8 implementation-bug narrative is a reproducibility detail. Move it to App E.
- L604 "Boundary of the architectural value proposition" is positioning (six differentiators). It repeats abstract L24 and is answered in App G, so it fits Discussion only if it cites measured receipts per differentiator; currently it cites none.

## 17. Smaller items
- tab:evidence-mapping caption (L545): "Cells without a measurable cycle-zero anchor are populated by the main-run readings" is stale. The table claims to map "every architectural claim" but has no rows for the baseline headline, the RUC ablation or contributions C1–C5.
- Holm family on the CHM axis is stated as m = 8 (L381, App D L96), but only 7 CHM contrasts exist per benchmark (B6 n/a). State m = 7 for CHM.
- No-reference labels: eq:mcnemar, sec:disc-limitations (the L624 signpost refers to "the limitations subsection" in words; use \Cref), sec:comp-references, sec:disc-headline, all B1–B8 subsubsections.
- Uninput auto tables (orphans; check whether intended): tab_tier_baseline_cost (placeholder), tab_phase1d_vs_phase1e_ruc (appears to duplicate tab_ruc_ablation), tab_headline, tab_headline_decomposition(_capability), tab_ces_per_system, tab_claim_evidence_map, tab_purity, tab_calibration, tab_continual, tab_cycle_progression, tab_grounding, tab_grounding_success_distribution, tab_per_tier_em_capability, tab_signal_fingerprint, tab_ablation_results, tab_audit_summary, fig_ustored_calibration.
- tab_baseline_pooled: "−11.0%" also appears as B2 at C5 and "+11.0%" as B6 EM at C2. Not an error, but quote "B1, C2" whenever −11.0% is cited.
