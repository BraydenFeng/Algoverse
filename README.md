# Desperation Circuit — Brayden's Gemma modules (v2)

Algoverse paper. Mechanistic account of hallucination in Gemma-2-9B-IT via the Ferrando unknown-entity SAE latent.

Active LaTeX: `raw/paper/proposal-v5.tex` (in the Vault). This repo executes the experiments described in §3 Methodology.

## Scope (v2 — workshop)

**Brayden owns all Gemma-2-9B-IT modules:**

1. **Emotion vector extraction** — 4 emotions (desperation, calm, sad, angry), 20 stories each, Claude Opus 4.7 generator
2. **Steering + capability check** — α sweep, MMLU drop ≤ 1pt
3. **Hallucination dose-response** — FaithEval unanswerable + orthogonal-projection ablation
4. **Mechanism tracing** — Imai 2010 causal mediation through Gemma Scope unknown-entity latent

**Out of scope (teammates):** Llama-3.1-8B-Instruct (behavioral only — no mediation per v2 §3).
**Deferred to v5:** Qwen-2.5-7B-Instruct, β-decomposition, held-out emotions, quantitative prediction.

## Execution environment

- VS Code locally (Windows) — edit code, write prereg notes, version control.
- Colab free T4 — sanity test on Gemma-2-2B-IT (smoke; $0).
- Colab Pro+ A100 / Vast.ai A100 80GB / Lambda A100 — Gemma-2-9B-IT real runs.
- HF Hub private dataset repo — stories + vectors + CSVs.

## Quick start

```bash
git clone https://github.com/<user>/desperation-circuit.git
cd desperation-circuit
bash scripts/setup_env.sh
```

Then open the notebook for the module you're running.

## Module status

| # | Module | Notebook | Status |
|---|--------|----------|--------|
| Sanity | Pipeline smoke on Gemma-2-2B-IT | `sanity_test.ipynb` | scaffold ready |
| 1 | Emotion vector extraction | `m1_extract.ipynb` | scaffold ready |
| 2 | Steering + MMLU | `m2_steer_mmlu.ipynb` | not yet written |
| 3 | FaithEval + orthogonal-projection ablation | `m3_faitheval.ipynb` | not yet written |
| 4 | Imai 2010 mediation | `m4_mediation.ipynb` | not yet written |

Modules written incrementally — don't write the next until the previous runs.
