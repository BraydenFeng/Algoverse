# Disclosures & methods log — desperation circuit (v2)

Running record of methodology decisions, fixes, and caveats that must be disclosed
in the paper or kept in mind. Last updated 2026-06-07.

## Environment / versions
- **M1–M3 ran on `transformers==4.57.6`; M4 runs on `transformers` 5.x** (required by
  `sae_lens` → `transformer_lens 3.3.0`, which pins `transformers>=5.4.0`). Disclose the
  split. M4 is internally consistent (all arms same version), so TE/ACME/ADE are valid;
  only the absolute baseline cross-reference to M3 carries the minor version difference.
- torch 2.8, CUDA 12.9, L40S (SageMaker `ml.g6e.xlarge`).

## Prompting (chat template)
- **M3 generation originally used RAW prompts (no chat template) — a bug.** Instruct
  models out-of-distribution: Llama rambled to the 128-token cap (~99% of outputs → judge);
  Gemma went **~20% empty at baseline** (40% at α=0.1, 50% at α=0.15).
- **Fixed:** apply each model's chat template (`use_chat_template=True`). Empties dropped to
  ~2%. The raw-prompt M3 results were **discarded and fully re-run** on the chat template.
  Commits: `c545cdc` (src), `d735814` (Gemma notebook).

## Classification (the desperate-refusal correction)
- **The rule classifier miscounted emotionally-phrased refusals as fabrications.** Under
  desperation steering the model produces refusals like *"I can't find it! I'm desperate!
  Please tell me!"* — genuine refusals the rule's abstention patterns missed, so they were
  labeled `fabricates`. The miscount **grows with α** (more steering → more desperate
  phrasing): correction was −2.3pt at α=0.3, −4.9pt at 0.5, **−17pt at 0.7**.
- **Fixed:** two-stage classifier — improved rule (abstention patterns now include "can't
  find", "no information", etc.; an answer-marker guard sends hedged "unknown but probably X"
  cases to the judge) + **LLM judge (Claude Haiku 4.5) on ambiguous only**. Report as
  "rule-based classification with LLM-judge fallback for ambiguous outputs."
- **PENDING — hand-label audit.** Validate the corrected classifier against ~100 hand-labels
  (weighted to high-α), target ≥85% agreement (project bar). Not yet done.
- Judge runs via Anthropic API. Bedrock was attempted but abandoned: Claude on Bedrock is
  billed via AWS Marketplace, which standard AWS promo credits don't cover, and access
  required FTU form + Marketplace perms.

## Capability gates
- **MMLU drop tolerance loosened from 1pt to 3pt** (PI-authorized). Justification to state:
  at n=1140, MMLU accuracy SE ≈ 1.4pt, so a 1pt gate sits *below* the measurement noise
  floor; 3pt ≈ 2 SE. Report drops with this framing, not as a binary pass/fail.
- **MMLU is insensitive to steering-induced generation degradation in Llama** — flat
  (~1.5–1.75pt drop) across α=0.3–0.7 while off_topic spiked to 10% at 0.7. Gemma's MMLU
  *does* respond (1.67pt@0.3 → 8.25pt@0.7). So **generation quality (off_topic) and MMLU
  measure different capability axes; report both.** Neither alone bounds the safe range.

## M3 results & caveats (corrected labels)
- **Effect is real and survived the re-classification.** Gemma **+15.3pt fabrication at
  α=0.5** (down from contaminated +20.1; ~75% survived), capability-preserving (MMLU
  −2.81pt, within loosened gate; off_topic flat). Llama **+6.5pt at 0.5**, off_topic flat
  to 0.6.
- **High-α reversal (Gemma):** corrected fabrication *peaks at α=0.5* then **declines at 0.7**
  (model reverts to desperate refusal). The raw "+25pt at 0.7" was almost entirely the
  labeling artifact. Honest claim: desperation drives fabrication up to α≈0.5, then defaults
  to (desperate) abstention.
- **Ablation is null in both models** — project-out shows no reduction below baseline →
  desperation is *sufficient* to drive fabrication but *not necessary* for baseline
  fabrication.
- **Model-dependence:** both replicate desperation→fabrication; Gemma stronger; failure modes
  differ (Llama loses coherence, Gemma loses reasoning while staying fluent).
- Gated (α≤0.1) effect is small (~+1–2pt); the strong effect lives at α=0.3–0.5 under the
  loosened gate.
- **Llama layer 21 is an unvalidated borrow** (no published Ferrando equivalent); flagged in
  config. A weak/different Llama effect could be partly a layer artifact.
- Final runs used **batch_size=1** (batched generation showed a small directional artifact;
  batch-1 matches the methodology).
- **PENDING:** paired/bootstrap CI on the α=0.5 effect; uniform re-classification of Llama
  (its labels were mostly clean but should match Gemma's pipeline).

## M4-specific (mediation)
- **`ALPHA_HEAD = 0.5`** for mediation steering (max capability-preserving effect for Gemma).
- **SAE:** `google/gemma-scope-9b-pt-res`, layer 21, width 16k, `average_l0_66` (config
  originally specified `average_l0_75`, which does not exist for L21/16k; 66 is the nearest
  available — commit fixing this).
- **Unknown-entity latent derived on the IT model**, not the PT base model (Ferrando derives
  on PT). Note the deviation.
- **Pile-frequency filter NOT applied** (Ferrando's entity-specificity filter skipped for v2;
  flagged in the latent JSON). Revisit if the latent fails downstream validation.
- **Arm D (suppress) differs from Imai-strict D** (per PI ask — clamps to a low value across
  prompts rather than per-prompt M(1)). Core TE/ACME/ADE use arms A/B/C only.
- **Separation-score gate:** if the derived latent's min-across-types separation < ~0.4, the
  latent is too weak and mediation is not meaningful — do not report. **The derived latent
  (idx 15674) scored 0.30 — BELOW this gate.** So any M4 null is confounded: cannot cleanly
  separate "effect genuinely not mediated by the unknown-entity signal" from "latent too weak
  to detect mediation." Human-owned framing call.
- **Mediator capture position fix.** The latent does NOT fire at the FaithEval decision token
  (last token) — capturing there gave mediator ≡ 0 (degenerate). It fires at the *entity*
  token. Fixed with `_entity_position()` (argmax of latent activation across the prompt;
  falls back to last token if the latent never fires). After the fix M(0) mean = 6.60
  (range 0–21.4) — a real, varying mediator.
- **M4 FULL-SAMPLE result (n=2492 — clean NULL mediation).** Refusal rates: A=0.639,
  B(steered)=0.490, C(rescue)=0.488, D(suppress, clamp=0.0)=0.640. Mediator means M(0)=8.30,
  M(1)=9.10. Decomposition (Y=refusal): **TE=−14.9pt, ACME=+0.2pt, ADE=−15.1pt** (closes);
  **proportion mediated ≈ 0%.** Effect is ~100% direct. Three independent signals:
  (1) **Pipeline validates** — the +15.1pt fabrication TE matches M3's corrected +15.3pt@α=0.5,
  so the null is not a broken setup. (2) ACME≈0 — restoring the latent to baseline recovered
  0.2 of 14.9pt. (3) **Arm D is the clincher** — clamping the latent to 0 (maximal intervention,
  8.3→0) left refusal unchanged (0.640 vs 0.639), so the latent is *causally inert* for
  fabrication, not merely hard to move. M(1)>M(0) again — steering nudges the latent up, never
  suppresses it → the "desperation fabricates by suppressing the unknown-entity signal"
  hypothesis is disconfirmed.
- **Framing (human-owned):** report as "desperation's effect on fabrication is direct; the
  candidate unknown-entity latent neither tracks the effect (ACME≈0) nor drives behavior under
  direct clamping (arm D), and its separation (0.30) is below our 0.40 gate." The arm-D
  behavioral fact is independent of what the 0.30-separation latent semantically represents,
  so the null is stronger than "inconclusive."

## Ownership note
- Brayden owns Llama M1–M3 (2026-05-29 reassignment) and re-took Gemma M4 (2026-06-04, after
  the assigned teammate went unresponsive; L40S compute makes it feasible).
