import argparse, os, random, logging, csv, time, torch
import numpy as np
from typing import List
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import HfDeepSpeedConfig
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
from utils.omnimedqkv import OmniMedVQA_Dataset
from datetime import datetime

from rl.embedder import MedClipEmbedder
from rl.rewards import build_rewards_v2, standardize_each_then_mix, group_norm_and_clip
from rl.roi import to_pil_list
from rl.utils import (
    split_prompt_from_sft_batch, per_token_logps, first_eos_mask, ensure_cogvlm_images
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grpo-v2")

# Reward terms of Eq. 5: answer accuracy and output format score the answer;
# noise-level prediction and visual consistency (VEC) score the visual behaviour.
# VEC is computed against the clean image while the policy generates from the
# noised embedding.
ALL_REWARD_NAMES = ["acc", "fmt", "noise_fmt", "noise_pred", "vec"]

def parse_args():
    p = argparse.ArgumentParser("Noise-aware GRPO")

    p.add_argument("--model_path", type=str, default="checkpoints/sft")
    p.add_argument("--dataset_path", type=str, default="data/OmniMedVQA-Train")
    p.add_argument("--save_path", type=str, default="checkpoints/nap")
    p.add_argument("--actor_lora_path", type=str, default=None)
    p.add_argument("--ds_config", type=str, default="finetune_demo/ds_config.yaml")
    p.add_argument("--torch_type", type=str, default="torch.bfloat16")

    p.add_argument("--max_input_len", type=int, default=640)
    p.add_argument("--max_output_len", type=int, default=320)
    p.add_argument("--train_dataset_rate", type=float, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--noise_rate", type=float, default=0.5,
                    help="noise rate passed to OmniMedVQA_Dataset")

    p.add_argument("--use_doctor_prompt", type=bool, default=True)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_target", type=eval, default=[
        "vision_expert_query_key_value", "vision_expert_dense",
        "language_expert_query_key_value", "language_expert_dense",
        "vision_mlp.gate_proj", "vision_mlp.up_proj", "vision_mlp.down_proj",
        "language_mlp.gate_proj", "language_mlp.up_proj", "language_mlp.down_proj",
    ])

    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--save_step", type=int, default=100)
    p.add_argument("--kl_coef", type=float, default=0.02)

    p.add_argument("--w_acc", type=float, default=0.35)
    p.add_argument("--w_fmt", type=float, default=0.1)
    p.add_argument("--w_noise_fmt", type=float, default=0.05)
    p.add_argument("--w_noise_pred", type=float, default=0.15)
    p.add_argument("--w_vec", type=float, default=0.35)

    p.add_argument("--visdep_sidecar", type=str, default="data/OmniMedVQA/iodep_scores.jsonl")
    p.add_argument("--visdep_min_score", type=float, default=0.35)
    p.add_argument("--visdep_weighting", action="store_true")
    p.add_argument("--dep_sidecar", type=str, default="data/OmniMedVQA/visdep_scores.jsonl")
    p.add_argument("--dep_min", type=float, default=0.0)
    p.add_argument("--keep_if_dep_missing", action="store_true")
    p.add_argument("--adv_clip", type=float, default=5.0)

    p.add_argument("--log_path", type=str, default="logs")
    p.add_argument("--acc_log_path", type=str, default="logs")
    p.add_argument("--vec_log_path", type=str, default="logs")
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    dtype = eval(args.torch_type)

    import yaml
    with open(args.ds_config, "r") as f:
        ds_cfg = yaml.safe_load(f)
    hf_ds_config = HfDeepSpeedConfig(ds_cfg)
    ds_plugin = DeepSpeedPlugin(hf_ds_config=hf_ds_config)
    accelerator = Accelerator(deepspeed_plugin=ds_plugin)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True)

    dataset = OmniMedVQA_Dataset(
        root_dir=args.dataset_path, tokenizer=tokenizer, model=base,
        torch_type=dtype, input_length=args.max_input_len,
        output_length=args.max_output_len,
        noise_rate=args.noise_rate,
        use_doctor_prompt=args.use_doctor_prompt,
        visdep_sidecar=args.visdep_sidecar,
        visdep_min_score=args.visdep_min_score,
        visdep_weighting=args.visdep_weighting,
        dep_sidecar=args.dep_sidecar, dep_min=args.dep_min,
        keep_if_dep_missing=args.keep_if_dep_missing,
    )

    total = len(dataset)
    train_size = int(args.train_dataset_rate * total)
    val_size = total - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    subset_indices = train_ds.indices
    scores = torch.tensor([dataset.samples[i]['visdep_score'] for i in subset_indices], dtype=torch.float32)
    weights = (0.5 + 0.5 * scores).pow(2.0)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(subset_indices), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=dataset.custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.custom_collate_fn)

    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target, inference_mode=False, modules_to_save=["linear_proj"],
    )
    model = get_peft_model(base, lcfg)
    if args.actor_lora_path:
        ckpt_name = os.path.basename(args.actor_lora_path.rstrip("/"))
        step_global = int(ckpt_name.split("_")[2])
        logger.info(f"Resumed from {args.actor_lora_path}, step={step_global}")
    else:
        step_global = 0

    logger.info(">> Loading MedCLIP embedder ...")
    embedder = MedClipEmbedder(); embedder.eval()
    logger.info(">> MedCLIP ready.")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optim, num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_loader) * args.num_epochs,
    )
    model, optim, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optim, train_loader, val_loader, lr_scheduler
    )
    os.makedirs(args.save_path, exist_ok=True)

    if args.actor_lora_path:
        accelerator.load_state(args.actor_lora_path)
        logger.info(f"Resumed from {args.actor_lora_path}, step={step_global}")

    def set_seed_time(step=0, i=0):
        seed = time.time_ns() ^ os.getpid() ^ (step * 0x9E3779B1) ^ (i * 0x85EBCA77)
        try:
            import torch.distributed as dist
            if dist.is_initialized(): seed ^= dist.get_rank() * 0xC2B2AE35
        except Exception: pass
        seed &= 0xFFFFFFFF
        random.seed(seed); np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    metrics_sum = defaultdict(float); metrics_n = 0
    ema = {}; EMA_BETA = 0.98
    def _ema(k, v):
        if v is None: return
        ema[k] = v if k not in ema else EMA_BETA * ema[k] + (1 - EMA_BETA) * v

    rw = {"acc": args.w_acc, "fmt": args.w_fmt, "noise_fmt": args.w_noise_fmt,
          "noise_pred": args.w_noise_pred, "vec": args.w_vec}

    csv_path = os.path.join(args.save_path, "train_metrics.csv")
    if accelerator.is_main_process and not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = [
                "wall_time", "epoch", "step_global",
                "loss_mean", "kl_mean", "r_mean_mean",
                "adv_mean_mean", "adv_std_mean",
            ]
            header += [f"parts_{n}_mean" for n in ALL_REWARD_NAMES]
            header += ["noise_ratio_mean"]
            header += ["loss_ema", "kl_ema", "r_mean_ema"]
            header += [f"parts_{n}_ema" for n in ALL_REWARD_NAMES]
            w.writerow(header)

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader, initial=step_global, desc=f"Epoch {epoch}")):
            step_global += 1

            prompts, prompt_attn, prompt_lens, images_raw, extra = split_prompt_from_sft_batch(batch, tokenizer)
            prompt_tti = extra.get("prompt_token_type_ids", None)
            if images_raw is not None and prompt_tti is None:
                raise RuntimeError("batch is missing token_type_ids")
            B, P = prompts.size(0), prompts.size(1)
            device = accelerator.device

            noise_levels_tensor = batch.get("noise_level", None)
            if noise_levels_tensor is not None:
                noise_levels = [int(nl) for nl in noise_levels_tensor]
            else:
                noise_levels = [0] * B

            if images_raw is None:
                images_for_gen = None
                tti_for_gen = torch.zeros_like(prompts, device=device)
                images_clean_pil = None
            else:
                images_for_gen = ensure_cogvlm_images(images_raw, device, dtype)
                tti_for_gen = prompt_tti.to(device)

                images_clean_pil = to_pil_list(images_raw)

            gen_inputs = {
                "input_ids": prompts.to(device),
                "token_type_ids": tti_for_gen,
                "attention_mask": prompt_attn.to(device),
                "images": images_for_gen,
            }
            gen_kwargs = {
                "max_new_tokens": args.max_output_len,
                "pad_token_id": tokenizer.pad_token_id or 0,
                "top_p": 0.95, "temperature": 1.0, "do_sample": True,
                "noise_level": noise_levels,
            }

            completions_ids = []
            with torch.no_grad():
                for i in range(args.k):
                    set_seed_time(step_global, i)
                    outputs = accelerator.unwrap_model(model).generate(**gen_inputs, **gen_kwargs)
                    completions_ids.append(outputs[:, P:])

            eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or tokenizer.pad_token_id or 0
            Lmax = max(c.size(1) for c in completions_ids) if completions_ids else 1
            comp_padded = []
            for c in completions_ids:
                if c.size(1) < Lmax:
                    comp_padded.append(torch.cat([c, c.new_full((c.size(0), Lmax - c.size(1)), eos_id)], dim=1))
                else:
                    comp_padded.append(c)
            comp_cat = torch.cat(comp_padded, dim=0)
            comp_mask = first_eos_mask(comp_cat, eos_id=eos_id).to(prompts.dtype)

            comp_texts: List[str] = [tokenizer.decode(row, skip_special_tokens=True).strip() for row in comp_cat]
            assert len(comp_texts) == B * args.k

            prompts_k = prompts.repeat(args.k, 1)
            prompt_attn_k = prompt_attn.repeat(args.k, 1)
            prompt_tti_k = (torch.zeros_like(prompts_k, device=device) if images_raw is None
                            else prompt_tti.repeat(args.k, 1).to(device))
            comp_tti = torch.zeros_like(comp_cat, device=device)
            pc_ids = torch.cat([prompts_k, comp_cat.to(device)], dim=1)
            pc_attn = torch.cat([prompt_attn_k.to(device), comp_mask.to(device)], dim=1)
            pc_tti = torch.cat([prompt_tti_k, comp_tti], dim=1)

            if images_for_gen is None:
                images_k = None
            elif isinstance(images_for_gen, list):
                images_k = []; [images_k.extend(images_for_gen) for _ in range(args.k)]
            else:
                images_k = [images_for_gen] * args.k

            start = max(P - 1, 0)
            logp_actor = per_token_logps(
                model, input_ids=pc_ids, attention_mask=pc_attn,
                images=images_k, use_cache=False, token_type_ids=pc_tti
            )[:, start:]

            with torch.no_grad():
                kl_coef = float(args.kl_coef)
                with accelerator.unwrap_model(model).disable_adapter():
                    logp_ref = per_token_logps(
                        model, input_ids=pc_ids, attention_mask=pc_attn,
                        images=images_k, use_cache=False, token_type_ids=pc_tti
                    )[:, start:]

            per_tok_kl = (torch.exp(logp_ref - logp_actor) - (logp_ref - logp_actor) - 1.0) if kl_coef > 0 else torch.zeros_like(logp_actor)

            gold_texts = batch.get("answer_text", None) or [""] * B
            question_id = batch.get("question_id", None) or [""] * B

            reward_dict = build_rewards_v2(
                pred_texts=comp_texts,
                gold_texts=gold_texts,
                question_id=question_id,
                true_noise_levels=noise_levels,
                embedder=embedder,
                images_clean=images_clean_pil,
                B=B, K=args.k, device=device,
                w_acc=args.w_acc, w_fmt=args.w_fmt,
                w_noise_fmt=args.w_noise_fmt, w_noise_pred=args.w_noise_pred,
                w_vec=args.w_vec,
                acc_log_path=args.acc_log_path,
                vec_log_path=args.vec_log_path,
            )

            _to_dev = lambda x: None if x is None else x.to(device)
            parts_input = {n: (_to_dev(reward_dict.get(n)), rw[n]) for n in ALL_REWARD_NAMES}
            rewards_mix, parts_log = standardize_each_then_mix(parts_input)
            advantages = group_norm_and_clip(rewards_mix, clip=args.adv_clip)
            adv_flat = advantages.reshape(B * args.k, 1)

            imp_weight = torch.exp(logp_actor - logp_actor.detach())
            per_token_loss = -(imp_weight * adv_flat - kl_coef * per_tok_kl)
            comp_mask_f = comp_mask.to(per_token_loss.dtype)
            loss = ((per_token_loss * comp_mask_f).sum(1) / (comp_mask_f.sum(1) + 1e-8)).mean()

            optim.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optim.step(); lr_scheduler.step()

            with torch.no_grad():
                mean_kl = (((per_tok_kl * comp_mask_f).sum(1) / (comp_mask_f.sum(1) + 1e-8)).mean().item() if kl_coef > 0 else 0.0)
                r_mean = rewards_mix.mean().item()
            adv_mean, adv_std = advantages.mean().item(), advantages.std().item()

            metrics_sum["loss"] += float(loss.item())
            metrics_sum["kl"] += float(mean_kl)
            metrics_sum["r_mean"] += float(r_mean)
            metrics_sum["adv_mean"] += float(adv_mean)
            metrics_sum["adv_std"] += float(adv_std)
            for n in ALL_REWARD_NAMES:
                metrics_sum[f"r_{n}"] += float(parts_log.get(n, 0.0))
            metrics_sum["noise_count"] += sum(1 for nl in noise_levels if nl > 0)
            metrics_sum["batch_size"] += B
            metrics_n += 1

            if step_global % args.save_step == 0 and step_global != 0:
                avg = {}
                for k, v in metrics_sum.items():
                    if k in ("noise_count", "batch_size"):
                        continue
                    avg[k] = v / max(metrics_n, 1)

                noise_ratio = metrics_sum["noise_count"] / max(metrics_sum["batch_size"], 1)

                for k, v in avg.items():
                    _ema(k, v)

                parts_str = " | ".join(
                    f"{n}={avg.get(f'r_{n}', 0):.6f}" for n in ALL_REWARD_NAMES
                )
                ema_parts_str = ", ".join(
                    f"{n}={ema.get(f'r_{n}', 0):.6f}" for n in ALL_REWARD_NAMES
                )

                logger.info(
                    f"[v2] epoch {epoch} step {step_global} "
                    f"| mean(loss)={avg['loss']:.6f} "
                    f"| mean(KL)={avg['kl']:.6f} (coef={kl_coef:.6f}) "
                    f"| mean(r_mean)={avg['r_mean']:.6f} "
                    f"| mean(adv_mean)={avg['adv_mean']:.3e} | mean(adv_std)={avg['adv_std']:.3e} "
                    f"| mean(parts: {parts_str}) "
                    f"| noise_ratio={noise_ratio:.2%} "
                    f"|| EMA(loss)={ema.get('loss', 0):.6f}, EMA(KL)={ema.get('kl', 0):.6f}, "
                    f"EMA(r_mean)={ema.get('r_mean', 0):.6f}, "
                    f"EMA(parts: {ema_parts_str})"
                )

                if accelerator.is_main_process:
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        row = [
                            time.time(), epoch, step_global,
                            avg["loss"], avg["kl"], avg["r_mean"],
                            avg["adv_mean"], avg["adv_std"],
                        ]
                        row += [avg.get(f"r_{n}", 0) for n in ALL_REWARD_NAMES]
                        row += [noise_ratio]
                        row += [
                            ema.get("loss", 0), ema.get("kl", 0), ema.get("r_mean", 0),
                        ]
                        row += [ema.get(f"r_{n}", 0) for n in ALL_REWARD_NAMES]
                        w.writerow(row)

                if args.log_path and accelerator.is_main_process:
                    try:
                        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                        with open(args.log_path, "a", encoding="utf-8") as f:
                            f.write(f"------------- {current_time} Step {step_global} -------------\n")
                            f.write(
                                f"mean(loss)={avg['loss']:.6f} | mean(KL)={avg['kl']:.6f} (coef={kl_coef:.6f}) "
                                f"| mean(r_mean)={avg['r_mean']:.6f} "
                                f"| mean(adv_mean)={avg['adv_mean']:.3e} | mean(adv_std)={avg['adv_std']:.3e}\n"
                            )
                            f.write(f"mean(parts): {parts_str}\n")
                            f.write(f"noise_ratio: {noise_ratio:.2%}\n")
                            f.write(
                                f"EMA: loss={ema.get('loss', 0):.6f}, KL={ema.get('kl', 0):.6f}, "
                                f"r_mean={ema.get('r_mean', 0):.6f}\n"
                            )
                            f.write(f"EMA(parts): {ema_parts_str}\n\n")
                    except Exception as e:
                        logger.warning(f"failed to write log: {e}")

                ckpt = os.path.join(args.save_path, f"checkpoint_step_{step_global}")
                os.makedirs(ckpt, exist_ok=True)
                model.save_pretrained(ckpt, safe_serialization=True)
                accelerator.save_state(ckpt)

                metrics_sum.clear()
                metrics_n = 0

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
