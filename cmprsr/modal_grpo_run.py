"""
Modal entrypoint for the CMPRSR two-sided length-reward GRPO experiment.

Ports the training/eval logic from cmprsr_final.ipynb to run remotely on
Modal instead of a local T4/V100. Differences vs. the notebook, specific to
running on billed-by-the-second cloud GPU time:

  1. No FAST_MODE/FULL_MODE split — single config sized for a fast run on
     an A100-40GB.
  2. Rollout generation is batched: `model.generate(num_return_sequences=G)`
     produces all G group members for a prompt in one call.
  3. Each arm trains ONCE, then is evaluated immediately on that same
     trained adapter — no retrain-from-scratch-before-eval (that was the
     bug in the original notebook that discarded half the step budget).
  4. Everything is captured: full per-rollout training logs (not just
     aggregated step stats), the actual eval compression texts, both
     trained LoRA adapters (as downloadable zips), and phase-by-phase
     timing/cost.

Usage:
    modal run modal_grpo_run.py
"""

import io
import json
import os
import tempfile
import time
import zipfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import modal

app = modal.App("cmprsr-grpo")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers>=4.45.0",
    "peft>=0.12.0",
    "bitsandbytes>=0.43.0",
    "datasets>=2.20.0",
    "accelerate>=0.33.0",
    "scipy>=1.11.0",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0",
    "numpy",
    "matplotlib",
)

with image.imports():
    import gc
    import re

    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        get_linear_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from scipy import stats

GPU_TYPE = "A100-40GB"  # default; override per-call with --gpu (see local_entrypoint)
GPU_PRICES = {
    # $/hr, from modal.com/pricing — verify before trusting for exact billing
    "T4": 0.59,
    "L4": 0.80,
    "A10": 1.10,
    "L40S": 1.95,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "RTX-PRO-6000": 3.03,
    "H100": 3.95,
    "H200": 4.54,
    "B200": 6.25,
    "B300": 7.10,
}
GPU_PRICE_PER_HOUR = GPU_PRICES[GPU_TYPE]


@dataclass
class Config:
    seed: int = 42

    model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    max_input_tokens: int = 200
    max_new_tokens: int = 128

    group_size: int = 8          # G=2 makes std/MAD degenerate; G>=6 is stable
    learning_rate: float = 5e-5
    num_steps: int = 40
    warmup_steps: int = 3
    grad_accum_steps: int = 4

    train_size: int = 60
    eval_size: int = 30

    r_T_min: float = 0.1
    r_T_max: float = 0.7

    lambda_two_sided: float = 2.0
    deadband_tol: float = 0.07
    kl_coeff: float = 0.01


# ---- Reward functions (identical to cmprsr_final.ipynb cell-7) ------------

def compute_compression_ratio(orig_tokens: int, comp_tokens: int) -> float:
    return comp_tokens / max(orig_tokens, 1)


def r_len_original(r_C: float, r_T: float) -> float:
    return 1.0 - max(0.0, r_C - r_T)


def r_len_two_sided(r_C: float, r_T: float, lam: float = 2.0) -> float:
    return 1.0 - max(0.0, r_C - r_T) - lam * max(0.0, r_T - r_C)


def r_len_deadband(r_C: float, r_T: float, lam: float = 2.0, tol: float = 0.07) -> float:
    """
    Two-sided reward with a flat deadband around the target: rollouts
    within `tol` of r_T all tie at 1.0 (no differentiating signal),
    reproducing the "tie -> no gradient pressure" condition that made the
    original reward's short side stable -- but now centered on the target
    instead of extending infinitely on the short side. Outside the
    deadband, penalize normally (long-side slope 1, short-side slope lam)
    so genuinely off-target rollouts still get real gradient signal.

    This targets the mechanism the two runs' rollout logs actually showed:
    under the plain two-sided reward, only ~18% of rollouts in a group tie
    at the max r_len (vs 73% under the original reward), so almost every
    rollout gets pushed somewhere -- constant pressure, not just pressure
    away from bad outcomes. Widening the tie back out near the target
    (rather than only far below it) should let good rollouts sit still.
    """
    diff = r_C - r_T
    if abs(diff) <= tol:
        return 1.0
    if diff > tol:
        return 1.0 - (diff - tol)
    return 1.0 - lam * (abs(diff) - tol)


def r_qual_number_retention(compression: str, orig_numbers: list) -> float:
    """
    F1 of original numbers vs. numbers present in the compression, plus a
    readability guard: F1 alone can't tell "kept the numbers in readable
    text" from "crammed the numbers into a run-on string with no spaces" —
    the first run confirmed the model finds the second one under RL
    pressure (e.g. "Darrell&Allen'sr8:11nowtotage162calAlln10yrs49"), since
    it scores identically to a normal sentence on number-match alone.
    """
    comp_nums = re.findall(r"[-+]?\d+(?:\.\d+)?", compression)
    if not orig_numbers:
        base = 1.0 if not comp_nums else 0.7
    else:
        orig_set, comp_set = set(orig_numbers), set(comp_nums)
        recall = sum(1 for n in orig_numbers if n in comp_set) / len(orig_numbers)
        precision = (sum(1 for n in comp_nums if n in orig_set) / len(comp_nums)) if comp_nums else 1.0
        base = 2 * recall * precision / (recall + precision + 1e-8)

    # Natural English runs roughly 1 space per 5-6 chars (density ~0.18).
    # Well below that on a non-trivial-length string suggests words were
    # crammed together rather than genuinely written tersely; short
    # legitimate answers ("15", "10 8 4 2 6") are exempted by the length gate.
    stripped = compression.strip()
    if len(stripped) >= 15:
        space_density = stripped.count(" ") / len(stripped)
        if space_density < 0.08:
            base *= 0.5

    return base


def robust_norm(values, floor: float = 0.05):
    """
    Group-relative normalization via median/MAD instead of mean/std.
    `floor` is a minimum MAD, not a tiny numerical epsilon: if every rollout
    in a group scores ~identically (e.g. all r_qual=0 because generation
    degenerated), a near-zero MAD would blow up the advantage on any tiny
    residual difference. Treating spreads below `floor` as `floor` caps
    that blowup instead of amplifying noise into a huge gradient step.
    """
    x = np.asarray(values, dtype=np.float64)
    med = np.median(x)
    mad = max(np.median(np.abs(x - med)), floor)
    return (x - med) / mad


def extract_numbers(text: str) -> list:
    return re.findall(r"[-+]?\d+(?:\.\d+)?", text)


@app.function(image=image, gpu=GPU_TYPE, timeout=90 * 60)
def run_experiment(
    config_overrides: dict | None = None,
    baseline_eval_override: list | None = None,
    baseline_history_override: list | None = None,
    proposed_reward_type: str = "two_sided",
    gpu_type_for_cost: str | None = None,
):
    """
    gpu_type_for_cost: which GPU this call is actually running on, for
    accurate cost-estimate labeling — pass this whenever the call used
    `.with_options(gpu=...)` to override the decorator's default, since the
    function has no built-in way to introspect that at runtime otherwise.

    proposed_reward_type: which r_len variant the Proposed arm trains with
    ("two_sided" or "deadband" — see reward functions above).

    baseline_eval_override / baseline_history_override: when set, skip
    training+evaluating the Baseline arm entirely and reuse these (loaded
    from a prior run's saved results) for the comparison/plot/stats. Use
    this for a follow-up that only changes the Proposed arm (e.g. testing
    a different lambda) — Baseline is unaffected by that change, so
    retraining it again is wasted GPU spend.
    """
    t_start = time.time()
    timing = {}
    active_gpu_type = gpu_type_for_cost or GPU_TYPE
    active_gpu_price = GPU_PRICES.get(active_gpu_type, GPU_PRICE_PER_HOUR)
    cfg = Config()
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(cfg, k, v)
    print(f"Config overrides applied: {config_overrides}" if config_overrides else "Full config (no overrides)")
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model loaded (4-bit QLoRA, {active_gpu_type}) — {trainable_params:,} trainable params")
    timing["model_load_s"] = time.time() - t_start

    def build_prompt(text, target_tokens):
        messages = [
            {"role": "system", "content": f"Compress to ~{target_tokens} tokens, keep all numbers."},
            {"role": "user", "content": f"Compress:\n{text}"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    t0 = time.time()
    ds = load_dataset("openai/gsm8k", "main")
    train_raw = ds["train"].shuffle(seed=cfg.seed).select(range(cfg.train_size))
    eval_raw = ds["test"].shuffle(seed=cfg.seed).select(range(cfg.eval_size))

    def preprocess(dataset):
        data = []
        for idx, ex in enumerate(dataset):
            q = ex["question"]
            toks = tokenizer.encode(q, add_special_tokens=False)
            if len(toks) < 20 or len(toks) > cfg.max_input_tokens - 60:
                continue
            rng = np.random.default_rng(cfg.seed * 1000 + idx)
            r_T = float(rng.uniform(cfg.r_T_min, cfg.r_T_max))
            target_toks = max(5, int(r_T * len(toks)))
            data.append(
                {
                    "idx": idx,
                    "question": q,
                    "orig_numbers": extract_numbers(q),
                    "prompt": build_prompt(q, target_toks),
                    "orig_len": len(toks),
                    "r_T": r_T,
                }
            )
        return data

    train_data = preprocess(train_raw)
    eval_data = preprocess(eval_raw)
    print(f"Train: {len(train_data)} examples | Eval: {len(eval_data)} examples")
    timing["dataset_prep_s"] = time.time() - t0

    @torch.no_grad()
    def generate_group(prompt: str, n: int, max_new: int, do_sample: bool):
        """Batched: n rollouts for one prompt in a single generate() call."""
        inp = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=cfg.max_input_tokens
        ).to(model.device)
        out = model.generate(
            **inp,
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=0.9 if do_sample else 1.0,
            num_return_sequences=n if do_sample else 1,
            pad_token_id=tokenizer.pad_token_id,
        )
        texts = tokenizer.batch_decode(out[:, inp.input_ids.shape[1] :], skip_special_tokens=True)
        return [t.strip() for t in texts]

    def compute_logprobs(prompt_text: str, response_text: str, use_adapter=True):
        full = tokenizer(prompt_text + response_text, return_tensors="pt").to(model.device)
        prompt_len = tokenizer(prompt_text, return_tensors="pt").input_ids.shape[1]
        ctx = model.disable_adapter() if not use_adapter else nullcontext()
        with ctx:
            logits = model(full.input_ids).logits
        log_probs = F.log_softmax(logits[0, prompt_len - 1 : -1], dim=-1)
        response_ids = full.input_ids[0, prompt_len:]
        return log_probs.gather(1, response_ids.unsqueeze(1)).squeeze()

    def train_step(examples, reward_type):
        """Returns (total_loss, rollouts) where each rollout carries full
        detail (text, per-component reward, advantage, source example idx)
        for logging — not just the aggregated stats needed for the loss.

        Generation runs in eval() mode, backward runs in train() mode.
        Sharing one mode across both is a real bug, not just extra noise:
        with gradient checkpointing on, KV caching is disabled, so every
        decoding step recomputes the full sequence from scratch — if the
        model is in train() mode during that, LoRA's dropout draws a fresh
        random mask on every single recompute, so already-generated tokens'
        hidden states shift under the model between decoding steps. That
        produces incoherent output even before any weight update, which is
        exactly what was observed (garbage text at step 0, loss in the
        millions from every rollout in a group collapsing to ~identical
        near-zero reward).
        """
        total_loss = 0.0
        all_rollouts = []

        for ex in examples:
            model.eval()
            texts = generate_group(ex["prompt"], cfg.group_size, cfg.max_new_tokens, do_sample=True)
            model.train()
            rollouts = []
            for comp in texts:
                comp_len = len(tokenizer.encode(comp, add_special_tokens=False))
                r_C = compute_compression_ratio(ex["orig_len"], comp_len)
                r_qual = r_qual_number_retention(comp, ex["orig_numbers"])
                if reward_type == "original":
                    r_len = r_len_original(r_C, ex["r_T"])
                elif reward_type == "two_sided":
                    r_len = r_len_two_sided(r_C, ex["r_T"], cfg.lambda_two_sided)
                elif reward_type == "deadband":
                    r_len = r_len_deadband(r_C, ex["r_T"], cfg.lambda_two_sided, cfg.deadband_tol)
                else:
                    raise ValueError(f"unknown reward_type: {reward_type}")
                rollouts.append(
                    {
                        "example_idx": ex["idx"],
                        "text": comp,
                        "r_qual": r_qual,
                        "r_len": r_len,
                        "r_C": r_C,
                        "delta": r_C - ex["r_T"],
                    }
                )

            if len(rollouts) < 2:
                continue

            adv_qual = robust_norm([r["r_qual"] for r in rollouts])
            adv_len = robust_norm([r["r_len"] for r in rollouts])
            adv = np.clip(1.0 * adv_qual + 0.5 * adv_len, -5.0, 5.0)

            for i, roll in enumerate(rollouts):
                roll["adv_qual"] = float(adv_qual[i])
                roll["adv_len"] = float(adv_len[i])
                roll["advantage"] = float(adv[i])

                lp_policy = compute_logprobs(ex["prompt"], roll["text"], use_adapter=True)
                lp_ref = compute_logprobs(ex["prompt"], roll["text"], use_adapter=False)
                pg_loss = -(lp_policy.mean()) * float(adv[i])
                kl_loss = (lp_policy - lp_ref).mean()
                loss = pg_loss + cfg.kl_coeff * kl_loss
                loss.backward()
                total_loss += loss.item()
                roll["loss"] = float(loss.item())

            all_rollouts.extend(rollouts)

        return total_loss, all_rollouts

    def train_arm(arm_name, reward_type):
        print(f"\n{'='*60}\nTraining: {arm_name} ({reward_type})\n{'='*60}")
        t_arm = time.time()

        for n, p in model.named_parameters():
            if "lora_B" in n:
                torch.nn.init.zeros_(p)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.num_steps)

        history = []
        rollout_log = []
        for step in range(cfg.num_steps):
            batch = [train_data[i % len(train_data)] for i in range(step, step + cfg.grad_accum_steps)]
            loss, rollouts = train_step(batch, reward_type)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            for r in rollouts:
                r["step"] = step
                r["arm"] = arm_name
            rollout_log.extend(rollouts)

            if rollouts:
                deltas = [r["delta"] for r in rollouts]
                history.append(
                    {
                        "step": step,
                        "loss": loss,
                        "mean_delta": float(np.mean(deltas)),
                        "abs_delta": float(np.mean(np.abs(deltas))),
                    }
                )
                if step % 5 == 0 or step == cfg.num_steps - 1:
                    print(
                        f"  Step {step:3d} | Δ_CR {history[-1]['mean_delta']:+.3f} "
                        f"(|Δ| {history[-1]['abs_delta']:.3f}) | loss {loss:.4f}"
                    )
                if step % 10 == 0:
                    sample = rollouts[0]
                    print(f"    sample: r_qual={sample['r_qual']:.2f} r_len={sample['r_len']:.2f} "
                          f"| {sample['text'][:100]!r}")

        gc.collect()
        torch.cuda.empty_cache()
        elapsed = time.time() - t_arm
        print(f"  ({arm_name} training took {elapsed/60:.1f} min)")
        return history, rollout_log, elapsed

    @torch.no_grad()
    def evaluate(data, tag):
        model.eval()
        results = []
        t_eval = time.time()
        print(f"\nEvaluating {tag}...")
        for ex in data:
            comp = generate_group(ex["prompt"], 1, cfg.max_new_tokens, do_sample=False)[0]
            comp_len = len(tokenizer.encode(comp, add_special_tokens=False))
            r_C = compute_compression_ratio(ex["orig_len"], comp_len)
            results.append(
                {
                    "idx": ex["idx"],
                    "question": ex["question"],
                    "target_r_T": ex["r_T"],
                    "compression": comp,
                    "r_C": r_C,
                    "delta": r_C - ex["r_T"],
                    "abs_delta": abs(r_C - ex["r_T"]),
                    "r_qual": r_qual_number_retention(comp, ex["orig_numbers"]),
                }
            )
        elapsed = time.time() - t_eval
        print(f"  ({tag} eval took {elapsed/60:.1f} min)")
        return results, elapsed

    def save_adapter_zip():
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname in os.listdir(d):
                    zf.write(os.path.join(d, fname), fname)
            return buf.getvalue()

    # Train each arm once, evaluate immediately, THEN save its adapter —
    # all before the next arm's training resets lora_B to zero. Skip
    # Baseline entirely if a prior run's results were passed in for reuse.
    if baseline_eval_override is not None:
        print("\nReusing saved Baseline eval/history — skipping Baseline training+eval this run")
        eval_baseline = baseline_eval_override
        history_baseline = baseline_history_override or []
        rollout_log_baseline = []
        t_train_baseline = 0.0
        t_eval_baseline = 0.0
        adapter_baseline_zip = None
    else:
        history_baseline, rollout_log_baseline, t_train_baseline = train_arm("Baseline", "original")
        eval_baseline, t_eval_baseline = evaluate(eval_data, "Baseline")
        adapter_baseline_zip = save_adapter_zip()

    history_proposed, rollout_log_proposed, t_train_proposed = train_arm("Proposed", proposed_reward_type)
    eval_proposed, t_eval_proposed = evaluate(eval_data, "Proposed")
    adapter_proposed_zip = save_adapter_zip()

    timing["train_baseline_s"] = t_train_baseline
    timing["eval_baseline_s"] = t_eval_baseline
    timing["train_proposed_s"] = t_train_proposed
    timing["eval_proposed_s"] = t_eval_proposed
    timing["total_s"] = time.time() - t_start
    timing["estimated_cost_usd"] = round(timing["total_s"] / 3600 * active_gpu_price, 4)
    timing["gpu_price_per_hour"] = active_gpu_price

    b_deltas = np.array([r["delta"] for r in eval_baseline])
    p_deltas = np.array([r["delta"] for r in eval_proposed])
    b_abs, p_abs = np.abs(b_deltas), np.abs(p_deltas)
    t_stat, p_val = stats.ttest_rel(b_abs, p_abs)

    summary = {
        "baseline_abs_delta_mean": float(b_abs.mean()),
        "baseline_abs_delta_std": float(b_abs.std()),
        "proposed_abs_delta_mean": float(p_abs.mean()),
        "proposed_abs_delta_std": float(p_abs.std()),
        "baseline_r_qual_mean": float(np.mean([r["r_qual"] for r in eval_baseline])),
        "proposed_r_qual_mean": float(np.mean([r["r_qual"] for r in eval_proposed])),
        "t_stat": float(t_stat),
        "p_val": float(p_val),
        "significant": bool(p_val < 0.05 and p_abs.mean() < b_abs.mean()),
    }

    print("\n" + "=" * 70)
    print("RESULTS (Paired Evaluation)")
    print("=" * 70)
    print(f"Baseline  |Δ_CR|: {summary['baseline_abs_delta_mean']:.4f} ± {summary['baseline_abs_delta_std']:.4f}"
          f" | r_qual={summary['baseline_r_qual_mean']:.3f}")
    print(f"Proposed  |Δ_CR|: {summary['proposed_abs_delta_mean']:.4f} ± {summary['proposed_abs_delta_std']:.4f}"
          f" | r_qual={summary['proposed_r_qual_mean']:.3f}")
    print(f"Paired t-test: t={t_stat:+.3f}, p={p_val:.4f}")
    print("Significant improvement ✓" if summary["significant"] else "No significant difference")
    print(f"\nTotal wall time: {timing['total_s']/60:.1f} min | Est. cost: ${timing['estimated_cost_usd']:.2f}"
          f" ({active_gpu_type} @ ${active_gpu_price}/hr)")

    # Plain-text side-by-side transcript for quick eyeballing
    transcript_lines = []
    for b, p in zip(eval_baseline, eval_proposed):
        transcript_lines.append(f"--- eval idx {b['idx']} (target r_T={b['target_r_T']:.2f}) ---")
        transcript_lines.append(f"Q: {b['question']}")
        transcript_lines.append(f"[Baseline] r_C={b['r_C']:.2f} Δ={b['delta']:+.2f} r_qual={b['r_qual']:.2f}")
        transcript_lines.append(f"  {b['compression']}")
        transcript_lines.append(f"[Proposed] r_C={p['r_C']:.2f} Δ={p['delta']:+.2f} r_qual={p['r_qual']:.2f}")
        transcript_lines.append(f"  {p['compression']}")
        transcript_lines.append("")
    eval_transcript = "\n".join(transcript_lines)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for hist, label, color in [
        (history_baseline, "Baseline", "blue"),
        (history_proposed, "Proposed", "green"),
    ]:
        steps = [h["step"] for h in hist]
        axes[0, 0].plot(steps, [h["mean_delta"] for h in hist], label=label, color=color)
        axes[0, 1].plot(steps, [h["abs_delta"] for h in hist], label=label, color=color)
    axes[0, 0].set_title("Training: Mean Δ_CR")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 1].set_title("Training: Mean |Δ_CR|")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].violinplot([b_deltas, p_deltas], showmeans=True)
    axes[1, 0].set_xticks([1, 2])
    axes[1, 0].set_xticklabels(["Baseline", "Proposed"])
    axes[1, 0].set_title(f"Eval Δ_CR (p={p_val:.3f})")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].scatter(b_abs, p_abs, alpha=0.6)
    lim = max(b_abs.max(), p_abs.max()) * 1.1
    axes[1, 1].plot([0, lim], [0, lim], "k--", alpha=0.5)
    axes[1, 1].set_xlabel("Baseline |Δ_CR|")
    axes[1, 1].set_ylabel("Proposed |Δ_CR|")
    axes[1, 1].set_title("Paired Comparison")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plot_png = buf.getvalue()

    return {
        "config": asdict(cfg),
        "gpu_type": active_gpu_type,
        "timing": timing,
        "summary": summary,
        "history_baseline": history_baseline,
        "history_proposed": history_proposed,
        "rollout_log_baseline": rollout_log_baseline,
        "rollout_log_proposed": rollout_log_proposed,
        "eval_baseline": eval_baseline,
        "eval_proposed": eval_proposed,
        "eval_transcript": eval_transcript,
        "plot_png": plot_png,
        "adapter_baseline_zip": adapter_baseline_zip,
        "adapter_proposed_zip": adapter_proposed_zip,
    }


SMOKE_OVERRIDES = {
    "num_steps": 1,
    "group_size": 2,
    "grad_accum_steps": 1,
    "train_size": 6,
    "eval_size": 3,
    "warmup_steps": 0,
    "max_new_tokens": 48,
}


FOLLOWUP_OVERRIDES = {
    "lambda_two_sided": 1.0,  # was 2.0 — the full run showed 2.0 overcorrects
                              # to consistently-too-long; testing a gentler,
                              # symmetric penalty
}


RECIPE_FIX_OVERRIDES = {
    # Three reward-shape variants (lambda=2.0, lambda=1.0, deadband) all
    # produced the same overshoot -- rollout-log evidence showed lambda
    # barely matters post-normalization, so the reward shape isn't the
    # issue. Re-reading the actual Cmprsr paper (Table 5 / Appendix B.1)
    # showed we never replicated their GRPO recipe: they fine-tune GRPO on
    # top of an SFT-warmed-up checkpoint, at lr=5e-6, over ~80k rollouts.
    # We trained from a raw instruct model at lr=5e-5 (10x higher) on
    # ~1,280 rollouts. This tests the two cheap-to-fix parts of that gap
    # (LR, training scale) without attempting the SFT stage itself.
    "learning_rate": 5e-6,
    "num_steps": 100,
    "warmup_steps": 10,  # keep ~10% warmup ratio matching the paper, scaled to num_steps
}


@app.local_entrypoint()
def main(smoke: bool = False, followup: bool = False, deadband: bool = False,
         recipe_fix: bool = False, gpu: str = ""):
    """
    modal run modal_grpo_run.py              -> full run (both arms)
    modal run modal_grpo_run.py --smoke      -> tiny sanity check (~1-2 min)
                                                 before spending the full run's budget
    modal run modal_grpo_run.py --followup   -> re-test with lambda=1.0 instead of
                                                 2.0. Reuses saved Baseline.
    modal run modal_grpo_run.py --deadband   -> the actual fix: r_len_deadband
                                                 instead of r_len_two_sided (flat/tied
                                                 zone within +/-deadband_tol of target,
                                                 real penalty only outside it). The
                                                 rollout-log analysis showed lambda
                                                 barely matters post-normalization —
                                                 this targets the mechanism that
                                                 actually drove the overshoot (loss of
                                                 ties near target, not penalty
                                                 magnitude). Reuses saved Baseline.
    modal run modal_grpo_run.py --recipe-fix -> lr=5e-6 (was 5e-5) + num_steps=100
                                                 (was 40), reward_type back to plain
                                                 two_sided (lambda=2.0). Tests whether
                                                 the overshoot across all 3 reward
                                                 variants was actually caused by never
                                                 matching the paper's LR/training-scale,
                                                 not by reward shape. Reuses saved
                                                 Baseline.
    --gpu L4|A10|A100-40GB|H100 -> override the GPU tier for this call only
                                    (e.g. `--smoke --gpu L4` to measure actual
                                    $/step on a cheaper card before committing
                                    a larger run to it).
    """
    real_modes = sum([followup, deadband, recipe_fix])
    if real_modes > 1:
        raise ValueError("--followup, --deadband, --recipe-fix are mutually exclusive")

    config_overrides = None
    baseline_eval_override = None
    baseline_history_override = None
    proposed_reward_type = "two_sided"
    prefix = ""

    if followup:
        config_overrides = dict(FOLLOWUP_OVERRIDES)
        prefix = "followup_"
    elif deadband:
        proposed_reward_type = "deadband"
        prefix = "deadband_"
    elif recipe_fix:
        config_overrides = dict(RECIPE_FIX_OVERRIDES)
        prefix = "recipe_fix_"

    if real_modes >= 1 and not smoke:
        # Baseline reuse is skipped under --smoke even if combined with a real
        # mode: the smoke config's tiny eval_size wouldn't match the saved
        # baseline's real eval_size, and ttest_rel requires equal-length
        # arrays -- smoke's job is just to sanity-check mechanics, not to
        # produce a real paired comparison, so it trains its own tiny baseline.
        with open("modal_run_results.json") as f:
            prior = json.load(f)
        baseline_eval_override = prior["eval_baseline"]
        baseline_history_override = prior["history_baseline"]
        print(f"Reusing Baseline from modal_run_results.json "
              f"({len(baseline_eval_override)} eval examples, "
              f"{len(baseline_history_override)} training steps)")
        if config_overrides:
            print(f"Overrides for this run: {config_overrides}")
        print(f"Proposed arm reward_type: {proposed_reward_type}")

    if smoke:
        merged = dict(config_overrides) if config_overrides else {}
        merged.update(SMOKE_OVERRIDES)  # smoke sizing always wins (tiny run)
        config_overrides = merged
        prefix = "smoke_" + prefix if prefix else "smoke_"

    fn = run_experiment.with_options(gpu=gpu) if gpu else run_experiment
    if gpu:
        print(f"GPU override: {gpu}")
    result = fn.remote(
        config_overrides=config_overrides,
        baseline_eval_override=baseline_eval_override,
        baseline_history_override=baseline_history_override,
        proposed_reward_type=proposed_reward_type,
        gpu_type_for_cost=gpu or None,
    )

    with open(f"{prefix}phase2_results.png", "wb") as f:
        f.write(result.pop("plot_png"))

    adapter_baseline_zip = result.pop("adapter_baseline_zip")
    if adapter_baseline_zip is not None:
        with open(f"{prefix}baseline_adapter.zip", "wb") as f:
            f.write(adapter_baseline_zip)

    with open(f"{prefix}proposed_adapter.zip", "wb") as f:
        f.write(result.pop("adapter_proposed_zip"))

    with open(f"{prefix}eval_transcript.txt", "w") as f:
        f.write(result.pop("eval_transcript"))

    with open(f"{prefix}modal_run_results.json", "w") as f:
        json.dump(result, f, indent=2)

    mode = "smoke test" if smoke else ("followup" if followup else ("deadband" if deadband else "full run"))
    print(f"\n✓ Saved ({mode}):")
    print(f"  {prefix}phase2_results.png     — training curves + eval distributions")
    print(f"  {prefix}modal_run_results.json — config, timing, summary stats, full rollout + eval logs")
    print(f"  {prefix}eval_transcript.txt    — human-readable side-by-side eval compressions")
    if adapter_baseline_zip is not None:
        print(f"  {prefix}baseline_adapter.zip / {prefix}proposed_adapter.zip — trained LoRA adapters")
    else:
        print(f"  {prefix}proposed_adapter.zip — trained LoRA adapter (baseline_adapter.zip reused from prior run)")
    print("\n" + json.dumps(result["summary"], indent=2))
    print(json.dumps(result["timing"], indent=2))
    if smoke:
        print("\nIf loss looks sane and eval_transcript.txt reads as coherent English, "
              "re-run without --smoke for the full experiment.")
