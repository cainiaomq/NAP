# -*- coding: utf-8 -*-

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import re
import time, random, argparse, logging
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, get_linear_schedule_with_warmup
from qwen_vl_utils import process_vision_info
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from tqdm import tqdm

from utils.omnimedqkv_lingshu import OmniMedVQA_Dataset_Lingshu
from rl.rewards import build_rewards_v2, standardize_each_then_mix, group_norm_and_clip
from rl.embedder import MedClipEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_REWARD_NAMES = ["acc", "fmt", "noise_fmt", "noise_pred", "vec"]
LEVEL_RATIO = {0: 0.0, 1: 0.072, 2: 0.144, 3: 0.216, 4: 0.288, 5: 0.360}


class VisualNoiseInjector:
    def __init__(self, visual_module):
        self.level = 0; self.enabled = True
        visual_module.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if not self.enabled or self.level <= 0:
            return out
        t = out[0] if isinstance(out, tuple) else out
        ratio = LEVEL_RATIO.get(int(self.level), 0.0)
        if ratio > 0:
            t.add_(torch.randn_like(t) * (ratio * t.float().std()).to(t.dtype))
        return out


def decide_noise_level(rate):
    return random.randint(1, 5) if random.random() < rate else 0


def build_prompt_inputs(sample, processor, device):
    msg = [{"role": "user", "content": [
        {"type": "image", "image": sample["image"]},
        {"type": "text", "text": sample["prompt_text"]}]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msg)
    inputs = processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(device)
    return inputs


def seq_logps(model, input_ids, attention_mask, pixel_values, image_grid_thw, prompt_len):
    
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw, use_cache=False)
    logits = out.logits[:, :-1, :]
    tgt = input_ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return tok_logp[:, prompt_len - 1:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="checkpoints/lingshu-st")
    p.add_argument("--dataset_path", default="data/OmniMedVQA-Train")
    p.add_argument("--save_path", default="checkpoints/lingshu-nap")
    p.add_argument("--visdep_sidecar", default="data/OmniMedVQA/iodep_scores.jsonl")
    p.add_argument("--dep_sidecar", default="data/OmniMedVQA/visdep_scores.jsonl")
    p.add_argument("--dep_min", type=float, default=0.0)
    p.add_argument("--keep_if_dep_missing", action="store_true")
    p.add_argument("--noise_rate", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--kl_coef", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_output_len", type=int, default=320)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--save_step", type=int, default=100)
    p.add_argument("--adv_clip", type=float, default=5.0)
    p.add_argument("--max_image_size", type=int, default=768)
    p.add_argument("--w_acc", type=float, default=0.35)
    p.add_argument("--w_fmt", type=float, default=0.1)
    p.add_argument("--w_noise_fmt", type=float, default=0.05)
    p.add_argument("--w_noise_pred", type=float, default=0.15)
    p.add_argument("--w_vec", type=float, default=0.35)
    p.add_argument("--visdep_min_score", type=float, default=0.35)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_target", type=eval,
                   default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    p.add_argument("--actor_lora_path", default=None)
    p.add_argument("--acc_log_path", default=None, help="per-sample accuracy reward log")
    p.add_argument("--vec_log_path", default=None, help="per-sample VEC reward log")
    p.add_argument("--local_rank", type=int, default=-1)
    args = p.parse_args()
    if args.acc_log_path is None:
        args.acc_log_path = os.path.join(args.save_path, "acc_reward.log")
    if args.vec_log_path is None:
        args.vec_log_path = os.path.join(args.save_path, "vec_reward.log")

    accelerator = Accelerator()
    device = accelerator.device
    processor = AutoProcessor.from_pretrained(args.model_path)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    injector = VisualNoiseInjector(base.visual)

    lcfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, target_modules=args.lora_target,
                      inference_mode=False, modules_to_save=["merger"])
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()

    logger.info(">> loading MedCLIP embedder ...")
    embedder = MedClipEmbedder(); embedder.eval()

    dataset = OmniMedVQA_Dataset_Lingshu(root_dir=args.dataset_path, max_image_size=args.max_image_size,
                                         use_doctor_prompt=True, visdep_sidecar=args.visdep_sidecar,
                                         visdep_min_score=args.visdep_min_score, dep_sidecar=args.dep_sidecar,
                                         dep_min=args.dep_min, keep_if_dep_missing=args.keep_if_dep_missing)
    n = len(dataset); nval = max(1, int(0.02 * n))
    train_ds, _ = random_split(dataset, [n - nval, nval], generator=torch.Generator().manual_seed(42))
    try:
        scores = torch.tensor([float(dataset.samples[i]["visdep_score"]) for i in train_ds.indices])
        weights = (0.5 + 0.5 * scores).pow(2.0)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds.indices), replacement=True)
        loader = DataLoader(train_ds, batch_size=1, sampler=sampler,
                            collate_fn=OmniMedVQA_Dataset_Lingshu.custom_collate_fn)
    except Exception as e:
        logger.warning(f"weighted sampling failed ({e}), falling back to shuffle")
        loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                            collate_fn=OmniMedVQA_Dataset_Lingshu.custom_collate_fn)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = get_linear_schedule_with_warmup(optim, args.warmup_steps, len(loader) * args.num_epochs)
    model, optim, loader, sched = accelerator.prepare(model, optim, loader, sched)
    resume_step = 0
    if args.actor_lora_path:
        accelerator.load_state(args.actor_lora_path)
        m = re.search(r"checkpoint_step_(\d+)", os.path.basename(args.actor_lora_path.rstrip("/")))
        resume_step = int(m.group(1)) if m else 0
        logger.info(f"resumed {args.actor_lora_path} (step_global={resume_step})")

    os.makedirs(args.save_path, exist_ok=True)
    rw = {"acc": args.w_acc, "fmt": args.w_fmt, "noise_fmt": args.w_noise_fmt,
          "noise_pred": args.w_noise_pred, "vec": args.w_vec}

    from collections import defaultdict
    import csv as _csv
    from datetime import datetime
    rewards_log_path = os.path.join(args.save_path, "rewards.log")
    csv_path = os.path.join(args.save_path, "train_metrics.csv")
    metrics_sum = defaultdict(float); metrics_n = 0
    ema = {}; EMA_BETA = 0.98
    def _ema(k, v):
        if v is None: return
        ema[k] = v if k not in ema else EMA_BETA * ema[k] + (1 - EMA_BETA) * v
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            header = ["wall_time", "epoch", "step_global", "loss_mean", "kl_mean", "r_mean_mean",
                      "adv_mean_mean", "adv_std_mean"]
            header += [f"parts_{n}_mean" for n in ALL_REWARD_NAMES] + ["noise_ratio_mean"]
            header += ["loss_ema", "kl_ema", "r_mean_ema"] + [f"parts_{n}_ema" for n in ALL_REWARD_NAMES]
            w.writerow(header)

    step_global = resume_step
    model.train()
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(tqdm(loader, desc=f"vdrfv2-ep{epoch}")):
            if epoch == 0 and step < resume_step:
                continue
            step_global += 1
            sample = {k: v[0] for k, v in batch.items()}
            level = decide_noise_level(args.noise_rate)
            gold_text = "<noise>L%d</noise> " % level + sample["answer_text"]
            qid = sample.get("question_id", "")
            image_clean = sample["image"]

            prompt_inputs = build_prompt_inputs(sample, processor, device)
            P = prompt_inputs["input_ids"].size(1)

            injector.level = level
            comp_texts, comp_ids_list = [], []
            with torch.no_grad():
                for i in range(args.k):
                    torch.manual_seed(time.time_ns() % (2**31) ^ (step_global * 131 + i))
                    g = accelerator.unwrap_model(model).generate(
                        **prompt_inputs, max_new_tokens=args.max_output_len,
                        do_sample=True, top_p=0.95, temperature=1.0)
                    comp = g[:, P:]
                    comp_ids_list.append(comp)
                    comp_texts.append(processor.batch_decode(comp, skip_special_tokens=True)[0].strip())

            reward_dict = build_rewards_v2(
                pred_texts=comp_texts, gold_texts=[gold_text], question_id=[qid],
                true_noise_levels=[level], embedder=embedder, images_clean=[image_clean],
                B=1, K=args.k, device=device,
                w_acc=args.w_acc, w_fmt=args.w_fmt, w_noise_fmt=args.w_noise_fmt,
                w_noise_pred=args.w_noise_pred, w_vec=args.w_vec,
                acc_log_path=args.acc_log_path, vec_log_path=args.vec_log_path)
            parts_input = {n: (None if reward_dict.get(n) is None else reward_dict[n].to(device), rw[n])
                           for n in ALL_REWARD_NAMES}
            rewards_mix, parts_log = standardize_each_then_mix(parts_input)
            advantages = group_norm_and_clip(rewards_mix, clip=args.adv_clip)

            injector.level = 0
            optim.zero_grad(set_to_none=True)
            loss_sum = 0.0; kl_sum = 0.0
            for kk, comp in enumerate(comp_ids_list):
                full = torch.cat([prompt_inputs["input_ids"], comp], dim=1)
                attn = torch.ones_like(full)
                mask_k = (comp != processor.tokenizer.pad_token_id).to(torch.float32)
                with torch.no_grad():
                    with accelerator.unwrap_model(model).disable_adapter():
                        lr_k = seq_logps(model, full, attn, prompt_inputs["pixel_values"],
                                         prompt_inputs["image_grid_thw"], P)
                la_k = seq_logps(model, full, attn, prompt_inputs["pixel_values"],
                                 prompt_inputs["image_grid_thw"], P)
                per_tok_kl_k = (torch.exp(lr_k - la_k) - (lr_k - la_k) - 1.0) if args.kl_coef > 0 else torch.zeros_like(la_k)
                imp_k = torch.exp(la_k - la_k.detach())
                adv_k = advantages[0, kk]
                per_tok_loss_k = -(imp_k * adv_k - args.kl_coef * per_tok_kl_k)
                loss_k = (per_tok_loss_k * mask_k).sum() / (mask_k.sum() + 1e-8)
                accelerator.backward(loss_k / args.k)
                loss_sum += float(loss_k.item()) / args.k
                kl_sum += float(((per_tok_kl_k * mask_k).sum() / (mask_k.sum() + 1e-8)).item()) / args.k
                del la_k, lr_k, per_tok_kl_k, per_tok_loss_k, loss_k
            optim.step(); sched.step()
            loss = torch.tensor(loss_sum); per_tok_kl = None; comp_mask = None

            mk = kl_sum
            r_mean = rewards_mix.mean().item()
            metrics_sum["loss"] += float(loss_sum)
            metrics_sum["kl"] += float(mk)
            metrics_sum["r_mean"] += float(r_mean)
            metrics_sum["adv_mean"] += float(advantages.mean().item())
            metrics_sum["adv_std"] += float(advantages.std().item())
            for n in ALL_REWARD_NAMES:
                metrics_sum[f"r_{n}"] += float(parts_log.get(n, 0.0))
            metrics_sum["noise_count"] += (1 if level > 0 else 0)
            metrics_sum["batch_size"] += 1
            metrics_n += 1

            if step_global % args.save_step == 0:
                avg = {k: v / max(metrics_n, 1) for k, v in metrics_sum.items()
                       if k not in ("noise_count", "batch_size")}
                noise_ratio = metrics_sum["noise_count"] / max(metrics_sum["batch_size"], 1)
                for k, v in avg.items():
                    _ema(k, v)
                ct = datetime.now().strftime("%d-%H-%M-%S-%f")
                parts_str = " | ".join(f"{n}={avg.get(f'r_{n}', 0):.6f}" for n in ALL_REWARD_NAMES)
                ema_parts_str = ", ".join(f"{n}={ema.get(f'r_{n}', 0):.6f}" for n in ALL_REWARD_NAMES)
                block = (
                    f"\n------------- {ct} Step {step_global} -------------\n"
                    f"mean(loss)={avg['loss']:.6f} | mean(KL)={avg['kl']:.6f} (coef={args.kl_coef:.6f}) "
                    f"| mean(r_mean)={avg['r_mean']:.6f} | mean(adv_mean)={avg['adv_mean']:.3e} "
                    f"| mean(adv_std)={avg['adv_std']:.3e}\n"
                    f"mean(parts): {parts_str}\n"
                    f"noise_ratio: {noise_ratio:.2%}\n"
                    f"EMA: loss={ema.get('loss',0):.6f}, KL={ema.get('kl',0):.6f}, r_mean={ema.get('r_mean',0):.6f}\n"
                    f"EMA(parts): {ema_parts_str}\n"
                )
                if accelerator.is_main_process:
                    with open(rewards_log_path, "a", encoding="utf-8") as f:
                        f.write(block)
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        w = _csv.writer(f)
                        row = [time.time(), epoch, step_global, avg["loss"], avg["kl"], avg["r_mean"],
                               avg["adv_mean"], avg["adv_std"]]
                        row += [avg.get(f"r_{n}", 0) for n in ALL_REWARD_NAMES] + [noise_ratio]
                        row += [ema.get("loss", 0), ema.get("kl", 0), ema.get("r_mean", 0)]
                        row += [ema.get(f"r_{n}", 0) for n in ALL_REWARD_NAMES]
                        w.writerow(row)
                logger.info(block.strip())
                metrics_sum.clear(); metrics_n = 0

                ck = os.path.join(args.save_path, f"checkpoint_step_{step_global}")
                os.makedirs(ck, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(ck, safe_serialization=True)
                accelerator.save_state(ck)
                logger.info(f"[ckpt] {ck}")

    final = os.path.join(args.save_path, "final_model")
    os.makedirs(final, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(final, safe_serialization=True)
    accelerator.save_state(final)
    logger.info(f"done -> {final}")


if __name__ == "__main__":
    main()
