# CMPRSR two-sided length reward — full experiment log

Follows up on [Cmprsr: Abstractive Token-Level Question-Agnostic Prompt Compressor](https://arxiv.org/abs/2511.12281).
Original hypothesis: the paper's length reward, `R_len = 1 - max(0, r_C - r_T)`, is flat
at 1.0 for any output shorter than target, giving GRPO no gradient signal to correct
under-compression. Everything below tests fixes for that.

## Files

- **`modal_grpo_run.py`** — the actual training/eval pipeline (runs on Modal, A100-40GB).
  `modal run modal_grpo_run.py [--followup | --deadband | --recipe-fix] [--smoke]`
- **`cmprsr_final.ipynb`** — reference implementation (reward math, dataset prep); not
  executed directly, kept in sync with the Modal script.
- **`results/`** — raw JSON output (config, full per-rollout logs, eval sets, timing/cost)
  and human-readable eval transcripts for each run.
- **`charts/`** — training curves + eval distributions per run, plus two summary charts
  comparing all of them.

## The story, short version

1. **First pass** (`baseline_and_lambda2_results.json`) found and fixed several real bugs
   in the original pipeline (summed reward before GRPO normalization, degenerate group
   size, an eval bug that retrained the model from scratch right before evaluating it,
   generation running in `train()` mode). Corrected, then tested the two-sided reward
   fix (`λ=2.0`): **statistically significant, but the wrong direction** — length-control
   error roughly doubled (`|Δ_CR|` 0.250 → 0.543, p=0.0065), with the model overshooting
   to consistently *too long* instead of hitting the target.

2. **`lambda1_results.json`** — tested a gentler `λ=1.0` on the theory the penalty was too
   aggressive. No change (0.562, still worse than baseline). Checking the actual
   per-rollout reward logs showed why: GRPO's advantage is normalized *within* each
   group, so it mostly reflects rank, not reward magnitude — `λ` barely functions as a
   tunable dial under this normalization.

3. **Deadband variant** (flat/no-penalty zone near the target, tested live, not saved —
   run was stopped once the training curve showed the same overshoot pattern by step 25).
   Reward *shape* wasn't the answer either.

4. Re-read the actual paper's training recipe (Table 5 / Appendix B.1): they fine-tune
   GRPO on top of an **SFT-warmed-up checkpoint**, at **lr=5e-6**, over **~80k rollouts**.
   This project trained GRPO from a raw instruct model at **lr=5e-5** (10x higher) on
   **~1,280 rollouts**, with no SFT stage at all.

5. **`recipe_fix_results.json`** — fixed the two cheap parts of that gap (lr → 5e-6,
   num_steps 40 → 100), kept the plain two-sided reward. **The overshoot disappeared**
   — length control came back to baseline level (`|Δ_CR|` 0.230 vs baseline's 0.250,
   p=0.612, no longer a regression). But quality (`r_qual`, number-retention F1) dropped
   to the lowest of any run (0.332 vs baseline's 0.450) — a real tradeoff, not a clean win.

## Bottom line

See `charts/summary_all_runs.png` and `charts/summary_quality_tradeoff.png`. None of the
three reward-shape variants tested (λ=2.0, λ=1.0, deadband) beat a correctly-evaluated
baseline on both length control and quality at once. Fixing the training recipe (LR,
scale) — not the reward math — resolved the systematic overshoot, but introduced a
quality regression instead. The likely next lever is the piece not attempted here: the
SFT warm-start stage, which is a larger undertaking than anything above and is left as
future work.
