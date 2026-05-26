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
│   ├── m3_retention.py        # M3 conditional/unconditional rate table (PI ask 2026-05-25)
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

- **Primary model:** `google/gemma-2-9b-it`, bf16, 42 layers, extract at layer 21 (Ferrando v2 Figure 9 peak for unknown-entity latent; was layer 28 pre-2026-05-25 layer move)
- **SAE (for M4):** `google/gemma-scope-9b-pt-res`, layer 21, width 16k, l0 ~75. PT (base-model) SAE per Ferrando §4 — entity-recognition latents are derived on PT, then steered into IT. Unknown-entity feature index not published in Appendix Q — will be re-derived via the separation-score recipe in M4. **TBD, blocks M4.**
- **Layer-suffixed artifact paths:** all M1/M2/M3 outputs now live under `outputs/{m1_vectors,m2,m3}/L{layer}/…`; HF Hub mirrors the same structure. Layer-28 artifacts from the original v2 run are preserved under `L28/` for comparison.
- **Emotions:** desperation, calm, sad, angry (LaTeX §3 paragraph 1 still says "loving, nostalgic" — stale)
- **Stories:** 20 per emotion, ~400 words, Claude Opus 4.7 generator, 20 narrative contexts (stratified)
- **Token skip:** 50 (Anthropic protocol)
- **PC project-out:** top PCs of neutral corpus explaining 50% variance
- **ℓ₂-normalize:** yes
- **Steering α-sweep:** [0.025, 0.05, 0.075, 0.1]
- **MMLU drop tolerance:** 1pt
- **FaithEval dataset:** `Salesforce/FaithEval-unanswerable-v1.0` (2,492 prompts)

## Open questions

1. **M4 SAE setup (partially resolved 2026-05-25).**
   - **SAE release locked:** `google/gemma-scope-9b-pt-res` (base model; entity-recognition latents transfer to IT per Ferrando §4). Was `gemma-scope-9b-it-res`.
   - **Layer locked:** 21. Brayden verified the Ferrando v2 Figure 9 peak directly. Same layer is now used for M1 extraction + M2 steering + M3 FaithEval + M4 mediation.
   - **Feature index still TBD.** Email-Ferrando route dropped (2026-05-25 Brayden); will be re-derived in M4 via the Section 4 separation-score recipe on Gemma-2-9B PT. This is the only remaining blocker for M4 to start.
   - **Triggers full M1→M2→M3 re-run at L21.** Existing L28 artifacts preserved under `outputs/{m1_vectors,m2,m3}/L28/` for comparison; L21 re-runs land in sibling `L21/` directories without overwriting.
2. **Repo sharing model with teammates** — Brayden's call; pending coordination with Akshat. Default: private GitHub, single shared repo, Llama on separate branch.
3. **Will Opus 4.7 stories be emotionally unambiguous by token 50?** Cell 4 of `sanity_test.ipynb` spot-checks two stories before paying for A100 extraction.

## M3 classifier validation (limitation — locked 2026-05-17)

100-item stratified hand-audit: 97% raw human–classifier agreement, but dominated by the tautological rule subset. Judge subset (~2% of items) too sparse (n=3) for a reliable estimate and shows the judge returning unparseable verdicts on hedged "context states X but not Y" outputs. **Decision (human-owned): report the directional M3 result with confidence (refusal suppressed, redistributed to off-topic, 4–10σ); treat precise fabricate/refuse magnitudes as classifier-limited; no classifier change, no judge-model swap.** Writeup limitation paragraph drafted (see session recap 2026-05-17). Judge-stratified re-audit deferred to future work — does not affect the qualitative finding.

## Decisions log

- **2026-05-10:** Scoped to v2 LaTeX. Stripped v5.2 over-build (m0_viability, prereg, Gemma-3, β-decomp, ROME control).
- **2026-05-10:** 4 emotions resolved as desperation + calm + sad + angry (control-emotion question, deferred since 2026-05-03).
- **2026-05-10:** Story count is Brayden's knob, not locked at LaTeX's 20.
- **2026-05-10:** Qwen probably deferred to v5 (Brayden flag); v2 LaTeX still lists three models but execution targets Gemma + Llama only.

## What artifacts live where

- **Stories:** `data/stories/{emotion}/{idx:03d}.txt` locally; push to HF Hub `BraydenF/desperation-circuit-artifacts/stories/` after M1
- **Vectors:** `outputs/m1_vectors/{emotion}.npy`; push to HF Hub `m1_vectors/`
- **FaithEval results:** `outputs/m{N}/faitheval_*.csv`; push to HF Hub `m{N}_results/`
- **M3 retention table:** `outputs/m3/retention_table.csv`; push to HF Hub `m3/retention_table.csv`. Per-arm conditional (on non-empty / strict-retained outputs) and unconditional (over all 2,492) refuse/fab rates with α-regime labels (≤ 0.3 interpretable, ≥ 0.5 overshoot). Built by Cell 12.
- **Code:** git, single private GitHub repo (pending push)
