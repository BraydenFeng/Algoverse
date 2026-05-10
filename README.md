# Desperation Circuit

Algoverse paper: mechanistic account of hallucination in Gemma-3-12B via the Ferrando unknown-entity SAE latent.

## Scope (this repo)

- **v2 workshop scope.** Gemma-3-12B PT only. 5 fitted emotions, no held-outs.
- **Brayden owns:** Modules 0.0, 0.25, 0.5.A, 0.5.C, 1, 1.5, 2.A, 2.B, 2.C.
- **Teammates own:** Llama replication (Module 2.F).
- **Deferred to v5.2 upgrade:** Modules 2.D (patching), 2.E (Pareto), 3 (held-out R²).

See `notebooks/` for module entry points and `prereg/` for the pre-registration doc.

## Execution environment

- Code edited locally in VS Code.
- Notebooks run on Colab Pro+ (short jobs) or rented A100 80GB on Vast.ai / Lambda (long jobs).
- Artifacts (vectors, CSVs, stories) pushed to a private HF Hub dataset repo, not git.

## Quick start (on Colab or rented box)

```bash
git clone https://github.com/<user>/desperation-circuit.git
cd desperation-circuit
bash scripts/setup_env.sh
```

Then open the notebook for the module you're running.

## Module status

| # | Module | Status |
|---|--------|--------|
| 0.0 | Base-model viability | not started |
| 0.25 | Pre-registration | not written |
| 0.5.A | ROME control | not started |
| 0.5.C | Ferrando construct validity on Gemma-3 | not started |
| 1 | Emotion vector extraction | not started |
| 1.5 | Validation battery | not started |
| 2.A | beta-decomposition | not started |
| 2.B | Ablation | not started |
| 2.C | Enhancement | not started |
