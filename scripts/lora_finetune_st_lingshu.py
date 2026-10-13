# -*- coding: utf-8 -*-

import os
import re
import random
import argparse
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, get_scheduler
from qwen_vl_utils import process_vision_info
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from tqdm import tqdm

from utils.omnimedqkv_lingshu import OmniMedVQA_Dataset_Lingshu
from rl.embedder import MedClipEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lingshu-curriculum-st")

LEVEL_RATIO = {0: 0.0, 1: 0.072, 2: 0.144, 3: 0.216, 4: 0.288, 5: 0.360}
_NOISE_TAG_RE = re.compile(r"<noise>.*?</noise>\s*", re.I | re.S)
_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Ea-e])")


def strip_noise_tags(s):
    return _NOISE_TAG_RE.sub("", s or "").strip()


def normalize_letter(t):
    m = _LETTER_RE.search(strip_noise_tags(t))
    return m.group(1).upper() if m else ""


class VisualNoiseInjector:
    def __init__(self, visual_module):
        self.level = 0
        visual_module.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.level <= 0:
            return out
        t = out[0] if isinstance(out, tuple) else out
        ratio = LEVEL_RATIO.get(int(self.level), 0.0)
        if ratio > 0:
            t.add_(torch.randn_like(t) * (ratio * t.float().std()).to(t.dtype))
        return out


@torch.no_grad()
def compute_vds(embedder, image, texts):
    
    img_emb = embedder.encode_image([image] * len(texts))
    txt_emb = embedder.encode_text(texts)
    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)
    return (img_emb * txt_emb).sum(-1).tolist()


def build_prompt_inputs(sample, processor, device):
    msg = [{"role": "user", "content": [
        {"type": "image", "image": sample["image"]},
        {"type": "text", "text": sample["prompt_text"]}]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msg)
    return processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(device)


def build_train_inputs(prompt_inputs, processor, answer_text, device):
    
    ans_ids = processor.tokenizer(answer_text, return_tensors="pt",
                                  add_special_tokens=False)["input_ids"].to(device)
    p_ids = prompt_inputs["input_ids"]
    full = torch.cat([p_ids, ans_ids], dim=1)
    labels = torch.full_like(full, -100)
    labels[:, p_ids.size(1):] = ans_ids
    return {
        "input_ids": full,
        "attention_mask": torch.ones_like(full),
        "pixel_values": prompt_inputs["pixel_values"],
        "image_grid_thw": prompt_inputs["image_grid_thw"],
        "labels": labels,
    }


def main():
    p = argparse.ArgumentParser("Lingshu curriculum self-distill ST")
    p.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    p.add_argument("--dataset_path", default="data/OmniMedVQA-Train")
    p.add_argument("--save_path", default="checkpoints/lingshu-st")
    p.add_argument("--visdep_sidecar", default="data/OmniMedVQA/iodep_scores.jsonl")
    p.add_argument("--visdep_min_score", type=float, default=0.35)
    p.add_argument("--dep_sidecar", default="data/OmniMedVQA/visdep_scores.jsonl")
    p.add_argument("--dep_min", type=float, default=0.0)
    p.add_argument("--keep_if_dep_missing", action="store_true")
    p.add_argument("--num_candidates", type=int, default=4)
    p.add_argument("--curriculum", default="0,1,2,3,4,5")
    p.add_argument("--gen_top_p", type=float, default=0.95)
    p.add_argument("--gen_temperature", type=float, default=1.0)
    p.add_argument("--gen_max_new_tokens", type=int, default=320)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--save_step", type=int, default=30, help="save every N batches")
    p.add_argument("--max_image_size", type=int, default=768)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_target", type=eval,
                   default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    p.add_argument("--resume_from", default=None)
    p.add_argument("--local_rank", type=int, default=-1)
    args = p.parse_args()

    curriculum = [int(x) for x in args.curriculum.split(",") if x.strip()]
    accelerator = Accelerator()
    device = accelerator.device
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    injector = VisualNoiseInjector(model.visual)

    lcfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_rank, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, target_modules=args.lora_target,
                      inference_mode=False, modules_to_save=["merger"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    logger.info(">> loading MedCLIP embedder ...")
    embedder = MedClipEmbedder(); embedder.eval()

    dataset = OmniMedVQA_Dataset_Lingshu(root_dir=args.dataset_path, max_image_size=args.max_image_size,
                                         use_doctor_prompt=True, visdep_sidecar=args.visdep_sidecar,
                                         visdep_min_score=args.visdep_min_score, dep_sidecar=args.dep_sidecar,
                                         dep_min=args.dep_min, keep_if_dep_missing=args.keep_if_dep_missing)
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        collate_fn=OmniMedVQA_Dataset_Lingshu.custom_collate_fn)

    optim = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)
    total_updates = len(loader) * args.num_epochs
    sched = get_scheduler("cosine", optimizer=optim, num_warmup_steps=args.warmup_steps,
                          num_training_steps=total_updates)
    model, optim, loader, sched = accelerator.prepare(model, optim, loader, sched)
    resume_epoch = 0; resume_step = 0
    if args.resume_from:
        accelerator.load_state(args.resume_from)
        m = re.search(r"checkpoint_epoch_(\d+)_step_(\d+)",
                      os.path.basename(args.resume_from.rstrip("/")))
        if m:
            resume_epoch, resume_step = int(m.group(1)), int(m.group(2))
        logger.info(f"resumed {args.resume_from} (epoch={resume_epoch}, step_global={resume_step})")

    os.makedirs(args.save_path, exist_ok=True)
    _fh = logging.FileHandler(os.path.join(args.save_path, "train.log"), encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_fh)
    import csv as _csv
    audit_path = os.path.join(args.save_path, "audit.jsonl")
    lvl_csv = os.path.join(args.save_path, "loss_per_level.csv")
    if accelerator.is_main_process and not os.path.exists(lvl_csv):
        with open(lvl_csv, "w", newline="", encoding="utf-8") as f:
            _csv.writer(f).writerow(["update_global", "epoch", "step_global", "noise_level", "mode", "loss"])

    step_global = resume_step; update_global = 0
    model.train()
    for epoch in range(args.num_epochs):
        if epoch < resume_epoch:
            continue
        for step, batch in enumerate(tqdm(loader, desc=f"ST-ep{epoch}")):
            if epoch == resume_epoch and step < resume_step:
                continue
            step_global += 1
            sample = {k: v[0] for k, v in batch.items()}
            gt_letter = normalize_letter(sample["answer_text"])
            image = sample["image"]

            prompt_inputs = build_prompt_inputs(sample, processor, device)
            P = prompt_inputs["input_ids"].size(1)

            injector.level = 0
            raw_cands = []
            with torch.no_grad():
                for _ in range(args.num_candidates):
                    g = accelerator.unwrap_model(model).generate(
                        **prompt_inputs, max_new_tokens=args.gen_max_new_tokens,
                        do_sample=True, top_p=args.gen_top_p, temperature=args.gen_temperature)
                    raw_cands.append(processor.batch_decode(g[:, P:], skip_special_tokens=True)[0].strip())
            clean_cands = [strip_noise_tags(t) for t in raw_cands]
            try:
                vds_list = compute_vds(embedder, image, clean_cands)
            except Exception as e:
                logger.warning(f"[vds] {e}"); vds_list = [0.0] * len(clean_cands)
            ok = [(c, v) for c, r, v in zip(clean_cands, raw_cands, vds_list)
                  if normalize_letter(r) == gt_letter and gt_letter != ""]

            if ok:
                chosen = max(ok, key=lambda x: x[1])[0]
                nl_pick = random.choice(curriculum)
                levels = [nl_pick]
                targets = {nl_pick: f"<noise>L{nl_pick}</noise> " + chosen.strip()}
            else:
                gt_with_letter = sample.get("answer_text", "")
                chosen = None
                levels = [0]
                targets = {0: "<noise>L0</noise> " + gt_with_letter.strip()}

            mode = "distill" if ok else "fallback"

            if accelerator.is_main_process:
                audit = {
                    "step_global": step_global, "mode": mode,
                    "curriculum_mode": "random_single",
                    "picked_level": levels[0],
                    "prompt_text": sample.get("prompt_text", ""),
                    "prompt_ntok": int(prompt_inputs["input_ids"].size(1)),
                    "gt_letter": gt_letter,
                    "candidates": [
                        {"raw": r, "clean": c, "vds": round(float(v), 4),
                         "pred_letter": normalize_letter(r),
                         "acc_ok": normalize_letter(r) == gt_letter and gt_letter != ""}
                        for r, c, v in zip(raw_cands, clean_cands, vds_list)
                    ],
                    "chosen_clean": chosen,
                    "fallback_gt": None if ok else sample.get("answer_text", ""),
                }
                with open(audit_path, "a", encoding="utf-8") as f:
                    import json as _json
                    f.write(_json.dumps(audit, ensure_ascii=False) + "\n")
            for nl in levels:
                injector.level = nl
                inp = build_train_inputs(prompt_inputs, processor, targets[nl], device)
                out = model(**inp)
                loss = out.loss
                optim.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                optim.step(); sched.step()
                update_global += 1
                if accelerator.is_main_process:
                    with open(lvl_csv, "a", newline="", encoding="utf-8") as f:
                        _csv.writer(f).writerow([update_global, epoch, step_global, nl, mode, float(loss.item())])

            if step_global % 20 == 0:
                logger.info(f"[ep{epoch} step{step_global} upd{update_global}] "
                            f"loss={loss.item():.4f} mode={'distill' if ok else 'fallback'} "
                            f"levels={levels} nCand_ok={len(ok)}")
            if step_global % args.save_step == 0:
                ck = os.path.join(args.save_path, f"checkpoint_epoch_{epoch}_step_{step_global}")
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
