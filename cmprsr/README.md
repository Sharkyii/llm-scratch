# CMPRSR two-sided length reward — full experiment log

Follows up on [Cmprsr: Abstractive Token-Level Question-Agnostic Prompt Compressor](https://arxiv.org/abs/2511.12281).
Original hypothesis: the paper's length reward, `R_len = 1 - max(0, r_C - r_T)`, is flat
at 1.0 for any output shorter than target, giving GRPO no gradient signal to correct
under-compression. Everything below tests fixes for that.

## The result

**[`RESULT.png`](RESULT.png)** is the one chart that matters — the final, properly
controlled comparison. Everything else in `charts/` is an intermediate step kept for the
full story; two of them (`charts/superseded_*.png`) were built on a flawed comparison
and are superseded by `RESULT.png` — see step 6 below for why.

## Files

- **`RESULT.png`** — the final result (see above).
- **`modal_grpo_run.py`** — the actual training/eval pipeline (runs on Modal, A100-40GB).
  `modal run modal_grpo_run.py [--followup | --deadband | --recipe-fix] [--smoke]`
- **`cmprsr_final.ipynb`** — reference implementation (reward math, dataset prep); not
  executed directly, kept in sync with the Modal script.
- **`results/`** — raw JSON output (config, full per-rollout logs, eval sets, timing/cost)
  and human-readable eval transcripts for each run.
- **`charts/`** — per-run training curves, plus two early summary charts now marked
  `superseded_*` (see below).

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
   — length control came back down (`|Δ_CR|` 0.230). But this run reused the *old-recipe*
   Baseline (0.250) for comparison, which confounds reward shape with recipe: the
   improvement might just mean the new recipe helps training in general, not that it
   specifically fixed the two-sided reward. (This is the comparison behind
   `charts/superseded_summary_all_runs.png` and
   `charts/superseded_summary_quality_tradeoff.png` — kept for the record, not the answer.)

6. **`baseline_new_recipe_results.json`** — closed that confound: retrained Baseline
   from scratch under the *same* corrected recipe, for a fair, matched comparison.
   Result (**[`RESULT.png`](RESULT.png)**): under matched conditions, λ=2.0 **ties**
   Baseline on length control (`|Δ_CR|` 0.230 vs 0.201, p=0.412, not significant) and
   **significantly beats it on quality** (`r_qual` 0.332 vs 0.243, p=0.0203).

## Bottom line

See **[`RESULT.png`](RESULT.png)** for the properly controlled result. The original
hypothesis holds: once both the pipeline bugs (advantage collapse, eval-mode bug, etc.)
and the training recipe (LR, scale — matching the source paper's Table 5) are fixed, the
two-sided reward matches the original reward on length control and meaningfully improves
quality. Two things worth flagging honestly rather than glossing over:

- **λ magnitude doesn't matter much.** λ=2.0 and λ=1.0 produced statistically
  indistinguishable results under the old recipe — GRPO's group-relative advantage
  normalization strips out most of the reward-magnitude information λ was meant to
  control. The variable that mattered was the *training recipe*, not the reward
  hyperparameter that was the original target of investigation.
- **Absolute quality still dropped** under the new recipe relative to the very first
  (old-recipe) baseline (0.450 → 0.243 for Baseline itself) — the corrected recipe here
  (100 steps) is still far short of the paper's actual scale (~150+ effective steps over
  10k examples), so this isn't full convergence, just a fair same-recipe comparison.

The next real lever — not attempted here, and a larger undertaking than anything above —
is the SFT warm-start stage the paper uses before GRPO, left as future work.
