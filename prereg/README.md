# Pre-registration

Module 0.25 of the v2 plan. Write and hash-commit this BEFORE any extraction code runs (deadline: May 15, 2026).

## What goes here

`v2_commitments.md` — the actual prereg doc. Must lock:

1. **5 fitted emotions** (resolve with Akshat before May 15)
2. **(valence, arousal, certainty) coordinates** for each emotion, with citations to published sources (Russell 1980 + Yik/Russell/Steiger 2011 + Smith & Ellsworth 1985 for certainty)
3. **Anthropic protocol parameters** (100 topics, 12 stories, token-50+, ~2/3 depth, PC project-out, ℓ₂-normalize) — already in `config.yaml`, lock here as source of truth
4. **FaithEval scoring rule** — 6 abstention labels; rule-based + Claude judge; hand-label 100 to validate
5. **v2 acceptance criteria:**
   - M2.B: (H_steered − H_ablated) / (H_steered − H_baseline) ≥ 0.30 AND overlap-neuron ablation ≥ 2× random-baseline ablation effect
   - M2.C: H_enhanced − H_baseline ≥ 0.30 × (H_steered − H_baseline)
6. **Steering magnitude** α = 0.05; document any relaxation
7. **Split-outcome framing** (workshop-only path; v5.2 conditional language)

## Procedure

1. Draft `v2_commitments.md` with all six co-authors' input
2. Commit to git
3. Note the commit hash in every paper draft
4. Reference: `prereg/v2_commitments_<hash>.md`

Acceptance: a third party could reproduce the experimental targets from this doc alone.
