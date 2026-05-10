# Desperation Circuit — Claude project instructions

Algoverse research paper code repo. Brayden owns all Gemma-2-9B-IT modules; teammates own Llama-3.1-8B-Instruct. v2 workshop scope only — no scope creep to v5.2/ICML without explicit approval.

## Scope discipline (non-negotiable)

**Approved (v2 LaTeX `raw/paper/proposal-v5.tex` §3):**
- Emotion vector extraction (4 emotions: desperation, calm, sad, angry)
- Steering + MMLU capability check
- FaithEval-unanswerable dose-response + orthogonal-projection ablation
- Imai 2010 causal mediation through unknown-entity SAE latent

**Not approved (do NOT build):**
- Module 0.0 viability check
- Module 0.25 pre-registration
- ROME factual-recall control
- β-decomposition
- Pareto frontier
- Curiosity / Contempt / random direction held-outs (radioactive during v2 — would compromise v5 optionality)
- Activation patching
- Gemma-3-12B (v2 uses Gemma-2-9B-IT)
- Gemma Scope 2 (v2 uses Gemma Scope original — what Ferrando used)
- Qwen-2.5-7B (probably deferred to v5; teammates' problem regardless)

If a new module or experiment idea comes up, flag it as "outside v2 LaTeX" and ask Brayden before building.

## Known facts

- **LaTeX §3 bug:** extraction paragraph still lists "loving, nostalgic" emotions. Controls description correctly lists "calm, sad, angry." Resolution is calm/sad/angry; LaTeX edit owned by Akshat / writing lead.
- **Story count is tunable** (`extraction.stories_per_emotion` in config.yaml). v2 says 20; Brayden controls. Don't argue for a number unless results in M1 indicate noise.
- **Ferrando feature index** for unknown-entity latent on Gemma-2-9B is in Ferrando v2 (arXiv:2411.14257) Appendix Q. Not yet pulled into config.yaml. M4 mediation can't run until that's resolved.
- **Module ownership:** Brayden = Gemma-2-9B-IT entirely. Llama = teammates (no mediation, just behavioral). If repo shared with team, Llama goes on a branch or fork.

## Code style (per global CLAUDE.md + project specifics)

- Tabs for indentation, double quotes, snake_case
- PEP 8 + FastAPI/Flask/Django conventions where they apply
- Comments only for non-obvious "why" — never narrate "what"
- Wrap I/O / network / parsing in try/except with informative messages (HF download failures should tell you to accept the model license)
- Use `src/lib/config.py` to read `config.yaml` — never hardcode model IDs or paths
- Notebooks are thin orchestrators; real logic lives in `src/`

## Execution environment

- Local Windows + VS Code: edit code, write prereg notes, version control. Can't run Gemma-2-9B (~18GB VRAM in bf16).
- Colab free T4: `sanity_test.ipynb` on Gemma-2-2B-IT smoke (~$0).
- Colab Pro+ A100 / Vast.ai / Lambda: `m1_extract.ipynb` and downstream (Gemma-2-9B-IT).
- HF Hub private dataset repo `bray/desperation-circuit-artifacts`: stories + vectors + CSVs. Don't commit these to git.

## Build incrementally

Don't pre-build modules ahead of their predecessors. Order:
1. Sanity test passes
2. M1 extraction produces clean vectors (pairwise cosines OK)
3. M2 steering + MMLU
4. M3 FaithEval + ablation
5. M4 Imai mediation

Each module writes outputs to `outputs/m{N}_*/`. Push to HF Hub after each module completes.

## Update primer after material code changes

`primer.md` mirrors current code state. Update it whenever:
- A new module is built
- A file's purpose changes
- An open question is resolved
- An external dependency changes (HF release, sae_lens, etc.)

Don't put dogfood reactions, speculation, or session narration in primer. State only.
