# Primer — desperation-circuit

Live state of the codebase. Update after material changes.

## Current state (2026-05-10)

**Stage:** v2 scaffolding complete; no compute spent yet.

**What runs end-to-end:** nothing yet — code written but unexecuted.

**Next compute action:** `sanity_test.ipynb` on Colab free T4 (~$0, ~10 min).

## Module status

| # | Module | Code | Run | Output |
|---|--------|------|-----|--------|
| Sanity | Gemma-2-2B-IT pipeline smoke | ✅ | ⬜ | — |
| 1 | Emotion vector extraction | ✅ | ⬜ | — |
| 2 | Steering + MMLU | ⬜ | ⬜ | — |
| 3 | FaithEval + orthogonal-projection ablation | ⬜ | ⬜ | — |
| 4 | Imai 2010 mediation | ⬜ | ⬜ | — |

## File layout

```
desperation-circuit/
├── CLAUDE.md                  # Claude project rules; scope discipline + facts
├── primer.md                  # this file
├── README.md                  # user-facing project description
├── config.yaml                # single source of truth for models, SAEs, emotions, paths
├── requirements.txt           # pinned deps
├── .gitignore                 # outputs/ + secrets ignored
├── scripts/
│   └── setup_env.sh           # Colab/Vast/Lambda bootstrap (idempotent)
├── src/
│   ├── faitheval_eval.py      # FaithEval dose-response driver (used by M0/M3)
│   ├── generate_stories.py    # Opus 4.7 story generator (used by M1)
│   ├── extract_vectors.py     # M1 extraction pipeline
│   └── lib/
│       ├── config.py          # config.yaml loader w/ env overrides
│       ├── model_load.py      # Gemma-2-9B-IT + 2B-IT loaders + chat template helper
│       ├── sae_load.py        # Gemma Scope (original) loader; M4 only
│       └── classifier.py      # refuses/fabricates/off_topic; rule + Claude judge
├── notebooks/
│   ├── sanity_test.ipynb      # Gemma-2-2B-IT smoke before paying for A100 hours
│   └── m1_extract.ipynb       # M1 orchestrator
├── data/                      # stories live here once generated; gitignored
└── outputs/                   # vectors + CSVs land here; gitignored
```

## Configuration locked

- **Primary model:** `google/gemma-2-9b-it`, bf16, 42 layers, extract at layer 28 (~2/3 depth)
- **SAE (for M4):** `gemma-scope-9b-it-res`, layer 28, width 16k, l0 ~75. Unknown-entity feature index from Ferrando v2 Appendix Q — **TBD, blocks M4**
- **Emotions:** desperation, calm, sad, angry (LaTeX §3 paragraph 1 still says "loving, nostalgic" — stale)
- **Stories:** 20 per emotion, ~400 words, Claude Opus 4.7 generator, 20 narrative contexts (stratified)
- **Token skip:** 50 (Anthropic protocol)
- **PC project-out:** top PCs of neutral corpus explaining 50% variance
- **ℓ₂-normalize:** yes
- **Steering α-sweep:** [0.025, 0.05, 0.075, 0.1]
- **MMLU drop tolerance:** 1pt
- **FaithEval dataset:** `Salesforce/FaithEval-unanswerable-v1.0` (2,492 prompts)

## Open questions

1. **M4 SAE setup needs revisiting before M4 starts.** From Ferrando v2 (arXiv:2411.14257) + their code repo (github.com/javiferran/sae_entities):
   - **SAE release in `config.yaml:19` is wrong.** Entity-recognition latents are computed on the base model (paper §4), then steered into the IT model. Switch `release` from `gemma-scope-9b-it-res` to `google/gemma-scope-9b-pt-res`. Width `16k` is correct.
   - **Layer choice (`config.yaml:20`) is probably wrong.** Ferrando's Figure 9 (Gemma-2-9B) shows separation-score peak in middle layers (~18–25) then plateau. Current `layer: 28` is past peak. Base-model SAEs exist on every layer (unlike IT, which is limited to 10/21/32), so layer is flexible — likely 21 is the right call to match Ferrando.
   - **Feature index is not published.** Paper doesn't list indices; repo computes them dynamically via `mech_interp/feature_analysis.py` → JSON outputs not committed. Three paths to resolve, in order of preference: (a) **email Javier Ferrando** at jferrandomonsonis@gmail.com asking for the `train_latents_layers_*` JSON outputs for Gemma-2-9B — 5-min ask, possible same-day reply, zero compute; (b) run their pipeline ourselves (~half-day A100); (c) re-derive in our own M4 code.
   - Not blocking M1/M2/M3.
2. **Repo sharing model with teammates** — Brayden's call; pending coordination with Akshat. Default: private GitHub, single shared repo, Llama on separate branch.
3. **Will Opus 4.7 stories be emotionally unambiguous by token 50?** Cell 4 of `sanity_test.ipynb` spot-checks two stories before paying for A100 extraction.

## Decisions log

- **2026-05-10:** Scoped to v2 LaTeX. Stripped v5.2 over-build (m0_viability, prereg, Gemma-3, β-decomp, ROME control).
- **2026-05-10:** 4 emotions resolved as desperation + calm + sad + angry (control-emotion question, deferred since 2026-05-03).
- **2026-05-10:** Story count is Brayden's knob, not locked at LaTeX's 20.
- **2026-05-10:** Qwen probably deferred to v5 (Brayden flag); v2 LaTeX still lists three models but execution targets Gemma + Llama only.

## What artifacts live where

- **Stories:** `data/stories/{emotion}/{idx:03d}.txt` locally; push to HF Hub `BraydenF/desperation-circuit-artifacts/stories/` after M1
- **Vectors:** `outputs/m1_vectors/{emotion}.npy`; push to HF Hub `m1_vectors/`
- **FaithEval results:** `outputs/m{N}/faitheval_*.csv`; push to HF Hub `m{N}_results/`
- **Code:** git, single private GitHub repo (pending push)
