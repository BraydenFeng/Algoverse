# Desperation-circuit v2 — paper outline + discrepancy register

Planning artifact for the v2 workshop paper. Target ~5 pages body + appendix.
Two models: **Gemma-2-9B-IT** (deep-dive, mechanistic) and **Llama-3.1-8B-Instruct**
(behavioral replication). All numbers cross-checked against `disclosures.md`,
`primer.md`, `config.yaml`, and `src/`.

---

## Part A — section-by-section outline

### Title
"Steering an emotional 'desperation' direction induces hallucination in
instruction-tuned LLMs — but not through the unknown-entity feature."
Honest framing: a positive behavioral result + a clean mechanistic null.

### 1. Abstract (~150 words)
- Extract a linear "desperation" direction (token-mean activation difference,
  controls calm/sad/angry) at ~2/3 depth; add to the residual stream during
  FaithEval-unanswerable inference.
- Headline: Gemma fabrication **+15.3pt at α=0.5** (0.3403 → 0.4932),
  capability-preserving (**MMLU −2.81pt**), non-monotonic (peaks at 0.5, reverts
  to refusal at 0.7).
- Llama replicates: **+6.5pt at α=0.5** (0.3997 → 0.4727), smaller, distinct
  failure mode (off_topic 2%→10% at α=0.7).
- Orthogonal-projection **ablation null** in both → desperation sufficient, not
  necessary.
- Causal mediation (Imai 2010) through the Ferrando unknown-entity latent
  (idx 15674) is **null**: TE=−14.9pt, **ACME≈+0.2pt**, ADE=−15.1pt → ~100% direct.

### 2. Introduction (~0.75 page)
- 2.1 Motivation: hallucination under "pressure"; emotional state as a steerable
  cause; instruction-tuned models as deployment-relevant.
- 2.2 The question: does an emotion direction causally increase fabrication on
  *unanswerable* questions (where correct = abstention)?
- 2.3 Contributions:
  1. Reproducible emotion-vector extraction + residual steering with three
     matched controls.
  2. FaithEval dose-response: desperation raises fabrication (Gemma +15.3pt,
     Llama +6.5pt) under an MMLU + off_topic gate.
  3. Non-monotonic curve + ablation null → sufficient but not necessary.
  4. Pre-registered causal-mediation test of the unknown-entity-suppression
     hypothesis — rejected (direct effect, latent inert under clamping).
  5. Cross-model replication with model-dependent failure modes.
- 2.4 Honesty preview: deep-dive (M1–M3) at one layer, mediation (M4) at another;
  why the mediation is still valid.

### 3. Related Work (~0.5 page)
- 3.1 Activation steering / linear representations (Anthropic emotion-vector
  protocol — token-skip-50 / PC-project-out).
- 3.2 Faithfulness & abstention (FaithEval-unanswerable, n=2492).
- 3.3 SAEs & entity-recognition features (Ferrando v2 unknown-entity latent,
  Gemma-Scope *original* not Scope-2).
- 3.4 Causal mediation in interpretability (Imai 2010; ACME/ADE; positioned as
  hypothesis-falsification).

### 4. Methods (~1.25 pages) — layer stated per module

**Table 1 (Methods overview):**

| Module | Model | Layer | n | Key config |
|---|---|---|---|---|
| M1 extraction | Gemma-2-9B-IT | **L28** | 20 stories/emotion | token_skip=50, PC-project-out 50% var, ℓ₂-norm |
| M2 steering+MMLU | Gemma-2-9B-IT | **L28** | 1140 MMLU | α∈{0.025…0.7}, gate 1pt→3pt |
| M3 FaithEval+ablation | Gemma-2-9B-IT | **L28** | 2492/arm | chat-template ON, batch=1 |
| M4 mediation | Gemma-2-9B-IT | **L21** | 2492 | Gemma-Scope L21 16k l0_66, latent 15674 |
| M1–M3 (Llama) | Llama-3.1-8B-Instruct | **L21** | 2492 | borrowed layer, unvalidated |

- 4.1 **M1 — extraction.** `src/extract_vectors.py`. Per-story residual capture,
  mean from token 50, per-emotion mean, subtract cross-emotion mean, project out
  neutral top PCs (50% var), ℓ₂-normalize. Gemma L28. Emotions = desperation +
  calm/sad/angry. Pairwise-cosine gate. Llama L21 cosines: −0.579 calm, −0.544
  sad, −0.100 angry.
- 4.2 **M2 — steering + MMLU.** `src/steering.py`. MMLU 20×57=1140, SE≈1.4pt.
  α = fraction of mean residual norm. Gate loosened 1pt→3pt. MMLU drops: Gemma
  1.67/2.81/8.25pt @ 0.3/0.5/0.7; Llama −1.75pt@0.5. L28 (Gemma), L21 (Llama).
- 4.3 **M3 — FaithEval + ablation.** `src/faitheval_eval.py`. Greedy,
  max_new_tokens=128, chat template ON, batch=1. Two-stage classifier (rule +
  Haiku judge) with desperate-refusal correction. Ablation = h − (h·v̂)v̂. L28
  (Gemma), L21 (Llama). n=2492/arm.
- 4.4 **M4 — Imai mediation (Gemma).** `src/m4_mediation.py`. Mediator =
  unknown-entity latent idx 15674, Gemma-Scope-9b-pt-res, L21, 16k, l0_66. Arms
  A/B/C/D (A baseline+capture, B steered+capture α=0.5, C rescue=steer+clamp to
  M(0), D suppress=clamp to 0). TE/ACME/ADE from A/B/C; D = inertness probe.
  **Mediator captured at the entity-token position** (`_entity_position()`;
  last-token capture read ≡0). L21 for injection + SAE + vector — internally
  self-consistent; different layer from M3 (explained in Limitations).

### 5. Results (~1.25 pages)
- 5.1 **M3 Gemma dose-response (Fig 1, Table 2).** baseline 0.3403/0.6380 →
  a0.05 0.3431/0.6324 → a0.1 0.3523/0.6204 → a0.15 0.3784/0.6035 → a0.2
  0.3848/0.5955 → a0.3 0.4057/0.5746 → **a0.5 0.4932/0.4904 (peak)** → a0.7
  0.4230/0.5690; ablated 0.3459/0.6300. off_topic flat 1.6–2.1%. +15.3pt headline
  + reversal. Ablation ≈ baseline → null.
- 5.2 **M2 capability gate (Fig 4).** MMLU vs α; α=0.5 within 3pt gate (−2.81pt);
  α=0.7 breaches (8.25pt). Report MMLU AND off_topic.
- 5.3 **M3 Llama replication (Fig 1 panel, Table 2).** baseline 0.3997/0.5835 →
  a0.5 0.4727/0.5084 → a0.7 0.5341/0.3632 (off_topic 0.1027). +6.5pt @ 0.5, MMLU
  −1.75pt. **Exclude α=0.6 (n=200 smoke).** Failure-mode contrast: Llama off_topic
  spikes to 10% at 0.7 vs Gemma stays fluent.
- 5.4 **M3 retention (Fig 5).** Conditional vs unconditional refuse/fab; regime
  labels (≤0.3 interpretable, ≥0.5 overshoot).
- 5.5 **M4 mediation (Fig 2 bars; Fig 6 mediator dist; Fig 7 separation; Table 3).**
  Arms A=0.639, B=0.490, C=0.488, D=0.640. M(0)=8.30, M(1)=9.10. **TE=−14.9pt,
  ACME=+0.2pt, ADE=−15.1pt**, ~0% mediated. Three signals: pipeline validates
  (+15.1≈+15.3); ACME≈0; arm-D inertness. M(1)>M(0) disconfirms suppression.
  Separation 0.30 noted.
- 5.6 **M1 vector geometry (Fig 3).** Within/between-emotion block structure.

### 6. Discussion (~0.5 page)
- Desperation sufficient not necessary (ablation null).
- Mechanism direct, not via the unknown-entity feature — narrows hypothesis space.
- Model-dependence: shared direction, different failure modes (Gemma loses
  reasoning while fluent; Llama loses coherence). MMLU misses generation collapse
  → report both.
- Non-monotonic reversal: high desperation → desperate abstention.

### 7. Limitations (~0.4 page)
- **Layer split (L28 deep-dive vs L21 mediation):** M4 internally consistent, TE
  replicates M3 across the gap, but not one seamless pipeline.
- **Llama L21 unvalidated borrow.**
- **Single-latent mediation; separation 0.30 < 0.40 gate** → "no evidence through
  *this specific feature*," not whole-circuit. Arm-D inertness makes the null
  robust to what the latent represents.
- **Classifier:** desperate-refusal correction grows with α; hand-label audit
  complete (directional finding solid, precise magnitudes classifier-limited).
- **No CIs yet** on α=0.5 → significance as effect-size + sigma, CIs deferred.
- **Version split:** transformers 4.57.6 (M1–M3) vs 5.x (M4).
- **SAE deviations:** latent on IT not PT; Pile filter skipped; arm D is global
  clamp (PI variant) not Imai-strict per-prompt.

### 8. Conclusion (~0.2 page)
Desperation steering causally induces hallucination (sufficient, not necessary;
capability-preserving at α=0.5), replicates across two families; the
unknown-entity feature is *not* the mediator. Future: whole-circuit mediation,
validated Llama layer, CIs.

### 9. Appendix
- A. Full per-α tables, both models.
- B. M2 full MMLU-per-α + per-emotion gate.
- C. M1 details: stories (20/emotion, Opus 4.7), neutral corpus, PCs, 4×4 cosine
  matrices.
- D. Classifier: rule patterns, judge prompt, hand-label audit + agreement,
  correction magnitudes per α.
- E. M4 full protocol: entity dataset (240 prompts), separation recipe, latent
  JSON, arm definitions, mediator-position fix, M(0)/M(1) dists, decomposition.
- F. Reproducibility: config snapshot, HF layout, version split, batch=1 note.
- G. Excluded/negative runs: α=0.6 Llama smoke (n=200), raw-prompt M3 (discarded).

---

## Part B — discrepancy & consistency register

Severity: **BLOCKER** (claim wrong / can't submit) · **FIX** (reconcile text
before submission) · **FOOTNOTE** (disclose only).

### Layer / pipeline coherence
1. **[FIX]** Gemma M3 at L28 but M4 at L21 — add Methods layer table + Limitations
   paragraph; frame M4's L21 TE (+15.1pt) as replicating M3's L28 effect
   (+15.3pt). Never imply one-layer pipeline.
2. **[FIX]** `config.yaml` says `extraction_layer: 21` but Gemma M3 artifacts live
   at `m3_results/L28/`. State that M3-Gemma numbers are the L28 artifacts.
3. **[FOOTNOTE]** Llama L21 is an unvalidated borrowed layer.
4. **[FOOTNOTE]** Cross-model replication compares L21-Llama to L28-Gemma — it's a
   *behavioral* replication, not layer-matched.

### Overreach / claims exceeding evidence
5. **[BLOCKER]** Don't claim "no mediation by the entity-recognition *circuit*" —
   one latent tested (0.30 separation). Restrict to "this specific latent."
6. **[BLOCKER]** No positive-mediation phrasing — ACME≈0 is a null.
7. **[FIX]** "Capability-preserving" scoped to α≤0.5 under the 3pt gate; pair with
   off_topic (Llama's 10% spike at 0.7 is MMLU-invisible).
8. **[FOOTNOTE]** State the non-monotonic reversal — don't show only the peak.
9. **[FOOTNOTE]** Use corrected +15.3pt, not raw +20.1/+25pt.

### Plan vs execution (vs CLAUDE.md scope)
10. **[FOOTNOTE]** Scope check passes — nothing "not approved" was run. No creep.
11. **[FOOTNOTE]** α-sweep ran to 0.5/0.7 but config lists only `[0.025…0.1]` —
    present the as-run sweep.
12. **[FOOTNOTE]** `config.yaml mmlu_drop_tolerance: 0.01` is stale (3pt used).
13. **[FOOTNOTE]** Author-contribution statement must reflect Brayden owns Llama
    M1–M3 + M4, not CLAUDE.md's stale "Llama = teammates."

### Stale LaTeX
14. **[FIX]** Proposal §3 still lists "loving, nostalgic" — resolved emotions are
    calm, sad, angry. Writing-lead owns the edit.

### Pending items blocking claims
15. **[FIX]** No CIs on the α=0.5 effect — compute bootstrap CIs (from existing
    CSVs) or phrase as effect-size + defer CIs.
16. **[FOOTNOTE]** Hand-label audit complete (supersedes CLAUDE.md "pending");
    keep distinct from the earlier weaker 100-item calibration.
17. **[FOOTNOTE]** Gemma corrected-label CSVs not yet on HF — upload before
    camera-ready.

### Code-vs-disclosure
18. **[DONE]** M4 capture position: committed `src/m4_mediation.py` now captures at
    the entity-token position (`_entity_position()`), matching the notebook run
    that produced M(0)=8.30. (Was previously last-token in the committed module.)
19. **[FOOTNOTE]** Cite M(0)=8.30 / M(1)=9.10 (full n=2492) consistently; the 6.60
    figure was a sub-sample sanity check.
20. **[FOOTNOTE]** `primer.md` still says SAE "l0 ~75"; actual is `l0_66`.
21. **[FOOTNOTE]** `faitheval_eval.py` docstring uses stale v5.2 module names
    (M2.B/M0.0) — cosmetic.

### Confirmed consistent (no action)
- M4 fab-TE +15.1 ≈ M3 corrected +15.3 @ α=0.5 → pipeline validation sound.
- Ablation null in both models.
- Chat-template fix applied in both Gemma and Llama M3; raw-prompt runs discarded.
- n=2492 across M3 arms and M4.
