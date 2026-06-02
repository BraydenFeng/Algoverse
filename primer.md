# Primer — desperation-circuit

Live state of the codebase. Update after material changes.

## Current state (2026-06-01)

**Stage:** Llama parallel track M1+M2 complete on SageMaker `ml.g6.xlarge` (L4 24GB); Gemma side still unrun. M3 Llama next.

**What runs end-to-end:** M1 Llama (vectors clean — desperation negatively correlates with all controls), M2 Llama (baseline MMLU 68.51% vs Llama's published ~68%; all 16 steered runs pass the 1pt capability gate).

**Next compute action:** `m3_faitheval_llama.ipynb` — FaithEval dose-response + ablation on L4. Tight on VRAM (~3 GB headroom over Llama-8B bf16); if OOM during generation, move to `ml.g6e.xlarge` (L40S 48GB) or drop to 4-bit.

## Module status

| # | Module | Code | Run | Output |
|---|--------|------|-----|--------|
| Sanity | Gemma-2-2B-IT pipeline smoke | ✅ | ⬜ | — |
| 1 | Emotion vector extraction (Gemma) | ✅ | ⬜ | — |
| 2 | Steering + MMLU (Gemma) | ⬜ | ⬜ | — |
| 3 | FaithEval + orthogonal-projection ablation (Gemma) | ⬜ | ⬜ | — |
| 4 | Imai 2010 mediation (Gemma — teammate-owned 2026-05-29) | ✅ | ⬜ | — |
| 1L | Emotion vector extraction (Llama-3.1-8B-Instruct) | ✅ | ✅ 2026-06-01 | `outputs/m1_vectors/llama_L21/` |
| 2L | Steering + MMLU (Llama) | ✅ | ✅ 2026-06-01 | `outputs/m2/llama_L21/` |
| 3L | FaithEval + orthogonal-projection ablation (Llama) | ✅ | ⬜ | — |

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
│   ├── m4_entity_dataset.py   # M4 hand-curated entity prompts (known/unknown × 4 types)
│   ├── m4_feature_derivation.py  # M4 separation-score recipe → unknown-entity latent index
│   ├── m4_mediation.py        # M4 Imai estimator + SAE clamp/capture hooks
│   └── lib/
│       ├── config.py          # config.yaml loader w/ env overrides
│       ├── model_load.py      # Gemma-2-9B-IT + 2B-IT loaders + chat template helper
│       ├── sae_load.py        # Gemma Scope (original) loader; M4 only
│       └── classifier.py      # refuses/fabricates/off_topic; rule + Claude judge
├── notebooks/
│   ├── sanity_test.ipynb               # Gemma-2-2B-IT smoke before paying for A100 hours
│   ├── m1_extract.ipynb                # M1 Gemma
│   ├── m2_steer_mmlu.ipynb             # M2 Gemma
│   ├── m3_faitheval.ipynb              # M3 Gemma
│   ├── m4_mediation.ipynb              # M4 Gemma (teammate-owned 2026-05-29)
│   ├── m1_extract_llama.ipynb          # M1 Llama-3.1-8B-Instruct parallel track
│   ├── m2_steer_mmlu_llama.ipynb       # M2 Llama
│   └── m3_faitheval_llama.ipynb        # M3 Llama (core protocol, no diagnostic add-ons)
├── data/                      # stories live here once generated; gitignored
└── outputs/                   # vectors + CSVs land here; gitignored
```

## Configuration locked

- **Primary model:** `google/gemma-2-9b-it`, bf16, 42 layers, extract at layer 21 (Ferrando v2 Figure 9 peak for unknown-entity latent; was layer 28 pre-2026-05-25 layer move)
- **SAE (for M4):** `google/gemma-scope-9b-pt-res`, layer 21, width 16k, l0 ~75. PT (base-model) SAE per Ferrando §4 — entity-recognition latents are derived on PT, then steered into IT. Unknown-entity feature index not published in Appendix Q — will be re-derived via the separation-score recipe in M4. **TBD, blocks M4.**
- **Layer-suffixed artifact paths:** all M1/M2/M3 outputs now live under `outputs/{m1_vectors,m2,m3}/L{layer}/…`; HF Hub mirrors the same structure. Layer-28 artifacts from the original v2 run are preserved under `L28/` for comparison.
- **Llama parallel track:** Llama-3.1-8B-Instruct config under `models.llama` (extraction_layer=21 — ~2/3 depth, working default). `load_llama()` in `src/lib/model_load.py`. `layer_suffix(cfg, "llama")` returns `llama_L21` so Llama outputs land in their own subdir (`outputs/{m1_vectors,m2,m3}/llama_L21/…` and HF Hub `m{N}_*/llama_L21/`) without colliding with Gemma artifacts. `extract_all(..., model_key="llama")` and `load_emotion_vector(emotion, model_key="llama")` route to the Llama namespace; the rest of the pipeline (`mmlu_eval`, `faitheval_eval`, hook factories) is model-agnostic and unchanged.
- **Emotions:** desperation, calm, sad, angry (LaTeX §3 paragraph 1 still says "loving, nostalgic" — stale)
- **Stories:** 20 per emotion, ~400 words, Claude Opus 4.7 generator, 20 narrative contexts (stratified)
- **Token skip:** 50 (Anthropic protocol)
- **PC project-out:** top PCs of neutral corpus explaining 50% variance
- **ℓ₂-normalize:** yes
- **Steering α-sweep:** [0.025, 0.05, 0.075, 0.1]
- **MMLU drop tolerance:** 1pt
- **FaithEval dataset:** `Salesforce/FaithEval-unanswerable-v1.0` (2,492 prompts)

## Open questions

1. **M4 SAE setup (resolved 2026-05-25, M4 scaffolded 2026-05-26).**
   - **SAE release locked:** `google/gemma-scope-9b-pt-res` (base model; entity-recognition latents transfer to IT per Ferrando §4).
   - **Layer locked:** 21. Same layer is used for M1 extraction + M2 steering + M3 FaithEval + M4 mediation.
   - **Feature index will be re-derived in M4 Cell 3.** Hand-curated entity dataset (30 known + 30 unknown per type × 4 types = 240 prompts) implements Ferrando §4 recipe; output JSON cached at `outputs/m4/L{layer}/unknown_entity_latents.json`. If separation score < 0.4 on first run, fall back to running Ferrando's full Wikidata pipeline (~half-day A100 extra).
   - **M4 scaffolding:** `src/m4_entity_dataset.py`, `src/m4_feature_derivation.py`, `src/m4_mediation.py`, `notebooks/m4_mediation.ipynb`. Four arms (baseline+capture, steered+capture, rescue, reverse). Headline α=0.3 (M3 L21 crossover). Cost ~90 min A100, ~$5-7 on Vast spot.
2. **Repo sharing model with teammates** — Brayden's call; pending coordination with Akshat. Default: private GitHub, single shared repo, Llama on separate branch.
3. **Will Opus 4.7 stories be emotionally unambiguous by token 50?** Cell 4 of `sanity_test.ipynb` spot-checks two stories before paying for A100 extraction.

## M3 classifier validation (limitation — locked 2026-05-17)

100-item stratified hand-audit: 97% raw human–classifier agreement, but dominated by the tautological rule subset. Judge subset (~2% of items) too sparse (n=3) for a reliable estimate and shows the judge returning unparseable verdicts on hedged "context states X but not Y" outputs. **Decision (human-owned): report the directional M3 result with confidence (refusal suppressed, redistributed to off-topic, 4–10σ); treat precise fabricate/refuse magnitudes as classifier-limited; no classifier change, no judge-model swap.** Writeup limitation paragraph drafted (see session recap 2026-05-17). Judge-stratified re-audit deferred to future work — does not affect the qualitative finding.

## Llama L21 results (2026-06-01)

**M1 — pairwise cosines between emotion vectors at L21:**

|             | desperation | calm   | sad    | angry  |
|-------------|-------------|--------|--------|--------|
| desperation | 1.000       | −0.579 | −0.544 | −0.100 |
| calm        | −0.579      | 1.000  | +0.271 | −0.566 |
| sad         | −0.544      | +0.271 | 1.000  | −0.437 |
| angry       | −0.100      | −0.566 | −0.437 | 1.000  |

Gate (|cos| with controls ≤ ~0.5 in the contaminating direction): passes — desperation is *negatively* correlated with all controls (correct: controls are well-separated, not contaminated). Layer-21 working default holds; no retune needed.

**M2 — capability gate:**

- Baseline MMLU: **0.6851** (781/1140; matches Llama-3.1-8B published ~0.68).
- ||h||@L21 = 20.92.
- All 16 (emotion × α) steered runs pass the 1pt drop tolerance. Per-emotion max usable α = 0.1 (top of the sweep) for every emotion. Largest observed drop = 0.79pt (calm @ α=0.1).
- Translation: capability gate is non-binding for Llama at this layer. M3 can run the full α-sweep at {0.025, 0.05, 0.075, 0.1}; α=0.1 is the headline dose.

**Infra notes (won't repeat):**

- `device_map="auto"` silently CPU/disk-offloaded layers under VRAM pressure on L4 → forward passes 30-100× slower. Fixed in `src/lib/model_load.py` by pinning to `{"": 0}` — fails clean on OOM instead.
- Two-token mode in `src/lib/model_load.py`: `HF_MODEL_TOKEN` (gated model download) + `HF_TOKEN` (artifact-repo writes). Used because the mentor accepted the Llama license but Brayden owns the artifact dataset.

## Decisions log

- **2026-05-10:** Scoped to v2 LaTeX. Stripped v5.2 over-build (m0_viability, prereg, Gemma-3, β-decomp, ROME control).
- **2026-05-10:** 4 emotions resolved as desperation + calm + sad + angry (control-emotion question, deferred since 2026-05-03).
- **2026-05-10:** Story count is Brayden's knob, not locked at LaTeX's 20.
- **2026-05-10:** Qwen probably deferred to v5 (Brayden flag); v2 LaTeX still lists three models but execution targets Gemma + Llama only.
- **2026-05-29:** Module reassignment. Brayden out of heavy compute → M4 (Gemma Imai mediation) handed to teammate; Llama-3.1-8B-Instruct M1–M3 (full parity, including orthogonal-projection ablation) taken on by Brayden. Llama notebooks added; CLAUDE.md ownership lines ("Llama = teammates") superseded for the M1–M3 scope.

## What artifacts live where

- **Stories:** `data/stories/{emotion}/{idx:03d}.txt` locally; push to HF Hub `BraydenF/desperation-circuit-artifacts/stories/` after M1
- **Vectors:** `outputs/m1_vectors/{emotion}.npy`; push to HF Hub `m1_vectors/`
- **FaithEval results:** `outputs/m{N}/faitheval_*.csv`; push to HF Hub `m{N}_results/`
- **M3 retention table:** `outputs/m3/retention_table.csv`; push to HF Hub `m3/retention_table.csv`. Per-arm conditional (on non-empty / strict-retained outputs) and unconditional (over all 2,492) refuse/fab rates with α-regime labels (≤ 0.3 interpretable, ≥ 0.5 overshoot). Built by Cell 12.
- **M4 mediation artifacts:** `outputs/m4/L{layer}/unknown_entity_latents.json` (top-N unknown-entity SAE latents from Ferrando §4 recipe), `outputs/m4/L{layer}/arm_{A,B,C,D}.csv` (per-prompt outputs + mediator values per arm), `outputs/m4/L{layer}/decision.txt` (Imai TE/ACME/ADE decomposition). Mirrored to HF Hub under `m4/L{layer}/`.
- **Code:** git, single private GitHub repo (pending push)
