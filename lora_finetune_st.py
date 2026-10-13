# -*- coding: utf-8 -*-


import argparse
import gc
import os
import random
import re
import json
import time
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

import yaml
import torch
from tqdm import tqdm
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import HfDeepSpeedConfig
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from torch.utils.tensorboard import SummaryWriter
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image

from utils.omnimedqkv import OmniMedVQA_Dataset
from utils.pmcvqa_st import PMC_VQA_Dataset
from utils.medxpertqa_st import MedXpertQA_Dataset
from rl.utils import split_prompt_from_sft_batch, ensure_cogvlm_images
from rl.embedder import MedClipEmbedder
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vdrfv2-curriculum-sft")


_LETTER_ANY_RE = re.compile(
    r'(?<![A-Za-z0-9Ａ-Ｚａ-ｚ０-９])'
    r'([A-EＡ-Ｅa-eａ-ｅ1-5１-５])'
)
_FULL_TO_HALF = str.maketrans('ａｂｃｄｅＡＢＣＤＥ１２３４５', 'abcdeABCDE12345')
_NUM_TO_LETTER = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E'}
_NOISE_TAG_RE = re.compile(r'<noise>.*?</noise>\s*', flags=re.IGNORECASE | re.DOTALL)


def strip_noise_tags(s: str) -> str:
    if not s:
        return ""
    return _NOISE_TAG_RE.sub('', s).strip()


def normalize_letter(txt: str) -> str:
    if not txt:
        return ""
    t = txt.strip().translate(_FULL_TO_HALF)
    m = _LETTER_ANY_RE.search(t)
    if not m:
        return ""
    ch = m.group(1).upper()
    if ch in _NUM_TO_LETTER:
        return _NUM_TO_LETTER[ch]
    if ch in ['A', 'B', 'C', 'D', 'E']:
        return ch
    return ""


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(-1)


@torch.no_grad()
def compute_vds_batch(embedder, images, texts: List[str]) -> List[float]:
    img_emb = embedder.encode_image(images)
    txt_emb = embedder.encode_text(texts)
    sim = _cos(img_emb, txt_emb)
    return sim.cpu().tolist()


def _chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def build_single_sample(dataset,
                        tokenizer,
                        image_pil: Image.Image,
                        question: str,
                        options: List,
                        label_text: str,
                        noise_level: int,
                        device,
                        dtype) -> Dict[str, torch.Tensor]:
    
    opt_text = "\n".join(f"{letter}. {opt}" for letter, opt in options)
    prefix1 = (dataset.doctor_prompt1 + "\n") if dataset.use_doctor_prompt else ""
    prefix2 = (dataset.doctor_prompt2 + "\n") if dataset.use_doctor_prompt else ""
    query = prefix1 + f"\nQuestion:\n{question}\nOptions:\n{opt_text}\nAnswer:\n" + prefix2

    input_data = dataset.model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        history=None,
        images=[image_pil],
        answer=label_text,
    )

    def pad_to_len(t, length, pad_value=0):
        l = len(t)
        return t[:length] if l >= length else torch.cat(
            (t, torch.full([length - l], fill_value=pad_value, dtype=t.dtype, device=t.device)), dim=0
        )

    max_length = dataset.max_length
    input_data['input_ids'] = pad_to_len(input_data['input_ids'], max_length, pad_value=128002)
    input_data['attention_mask'] = pad_to_len(input_data['attention_mask'], max_length, pad_value=0)
    input_data['token_type_ids'] = pad_to_len(input_data['token_type_ids'], max_length, pad_value=0)
    input_data['labels'] = pad_to_len(input_data['labels'], max_length, pad_value=-100)
    input_data['noise_level'] = torch.tensor(noise_level, dtype=torch.long)

    for key in input_data:
        if key == 'images':
            input_data[key] = [img.to(device).to(dtype) for img in input_data[key]]
        elif isinstance(input_data[key], torch.Tensor):
            input_data[key] = input_data[key].to(device)
    return input_data

def build_single_sample_fallback(dataset,
                        tokenizer,
                        image_pil: Image.Image,
                        question: str,
                        options: List,
                        label_text: str,
                        noise_level: int,
                        device,
                        dtype) -> Dict[str, torch.Tensor]:
    opt_text = "\n".join(f"{letter}. {opt}" for letter, opt in options)
    query = f"Question:\n{question}\nOptions:\n{opt_text}\nAnswer:\n"

    input_data = dataset.model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        history=None,
        images=[image_pil],
        answer=label_text,
    )

    def pad_to_len(t, length, pad_value=0):
        l = len(t)
        return t[:length] if l >= length else torch.cat(
            (t, torch.full([length - l], fill_value=pad_value, dtype=t.dtype, device=t.device)), dim=0
        )

    max_length = dataset.max_length
    input_data['input_ids'] = pad_to_len(input_data['input_ids'], max_length, pad_value=128002)
    input_data['attention_mask'] = pad_to_len(input_data['attention_mask'], max_length, pad_value=0)
    input_data['token_type_ids'] = pad_to_len(input_data['token_type_ids'], max_length, pad_value=0)
    input_data['labels'] = pad_to_len(input_data['labels'], max_length, pad_value=-100)
    input_data['noise_level'] = torch.tensor(noise_level, dtype=torch.long)

    for key in input_data:
        if key == 'images':
            input_data[key] = [img.to(device).to(dtype) for img in input_data[key]]
        elif isinstance(input_data[key], torch.Tensor):
            input_data[key] = input_data[key].to(device)
    return input_data

def collate_samples(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    out = {}
    keys = samples[0].keys()
    for k in keys:
        if k == 'images':
            out[k] = [s[k] for s in samples]
        elif isinstance(samples[0][k], torch.Tensor):
            out[k] = torch.stack([s[k] for s in samples], dim=0)
        else:
            out[k] = [s[k] for s in samples]
    return out


# Self-training of the noise annotation: sample K candidates per question, keep
# the correct one with the highest visual dependency score (VDS), or fall back to
# the ground-truth answer if none is correct. The retained text is then paired
# with a randomly drawn noise level.
@torch.no_grad()
def generate_and_filter_for_batch(
    model,
    tokenizer,
    dataset,
    embedder,
    items: List[Dict[str, Any]],
    num_candidates: int,
    max_new_tokens: int,
    top_p: float,
    temperature: float,
    micro_batch_gen: int,
    device,
    dtype,
    use_vds: bool = True,
) -> List[Dict[str, Any]]:
    sample_dicts: List[Optional[Dict[str, torch.Tensor]]] = []
    per_item_status: List[str] = []
    for it in items:
        try:
            sd = build_single_sample(
                dataset, tokenizer,
                it['image_pil'], it['question'], it['options'],
                label_text="placeholder",
                noise_level=0,
                device=device, dtype=dtype,
            )
            sample_dicts.append(sd)
            per_item_status.append("ok")
        except Exception as e:
            logger.warning(f"[build_sample] qid={it.get('qid')} failed: {e}")
            sample_dicts.append(None)
            per_item_status.append("skipped_gen_failed")

    valid_idx_in_items = [i for i, s in enumerate(per_item_status) if s == "ok"]
    valid_sample_dicts = [sample_dicts[i] for i in valid_idx_in_items]

    if len(valid_sample_dicts) == 0:
        return [
            {"status": st, "candidates": [], "chosen_clean_text": None}
            for st in per_item_status
        ]

    full_batch = collate_samples(valid_sample_dicts)
    prompts, prompt_attn, _, images_raw, extra = split_prompt_from_sft_batch(
        full_batch, tokenizer
    )

    if images_raw is None:
        for i in valid_idx_in_items:
            per_item_status[i] = "skipped_no_image"
        return [
            {"status": st, "candidates": [], "chosen_clean_text": None}
            for st in per_item_status
        ]

    prompt_tti = extra["prompt_token_type_ids"]

    K = num_candidates
    prompts_k = prompts.repeat_interleave(K, dim=0)
    prompt_attn_k = prompt_attn.repeat_interleave(K, dim=0)
    prompt_tti_k = prompt_tti.repeat_interleave(K, dim=0)

    images_k: List[Any] = []
    for img in images_raw:
        images_k.extend([img] * K)

    flat_meta: List[tuple] = []
    for local_i in range(len(valid_idx_in_items)):
        items_idx = valid_idx_in_items[local_i]
        for k in range(K):
            flat_meta.append((items_idx, k))

    BK = prompts_k.shape[0]
    assert BK == len(flat_meta), f"BK mismatch: {BK} vs {len(flat_meta)}"

    all_pred_texts: List[Optional[str]] = [None] * BK
    for chunk_indices in _chunks(list(range(BK)), micro_batch_gen):
        idx = torch.tensor(chunk_indices, dtype=torch.long, device=prompts_k.device)
        ps = prompts_k.index_select(0, idx).to(device)
        att = prompt_attn_k.index_select(0, idx).to(device)
        tti = prompt_tti_k.index_select(0, idx).to(device)

        imgs_chunk = []
        for i in chunk_indices:
            imgs_chunk.extend(images_k[i])
        imgs_for_gen = ensure_cogvlm_images(imgs_chunk, device, dtype)
        nl_list = [0] * len(chunk_indices)

        try:
            out = model.generate(
                input_ids=ps, token_type_ids=tti, attention_mask=att,
                images=imgs_for_gen,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or 0,
                top_p=top_p, temperature=temperature,
                do_sample=True,
                noise_level=nl_list,
            )
            P = ps.shape[1]
            comp_ids = out[:, P:]
            decoded = [tokenizer.decode(x, skip_special_tokens=True).strip() for x in comp_ids]
            for local_i, global_i in enumerate(chunk_indices):
                all_pred_texts[global_i] = decoded[local_i]
        except Exception as e:
            logger.warning(f"[gen] micro batch failed: {e}")
            for global_i in chunk_indices:
                all_pred_texts[global_i] = ""
        finally:
            del ps, att, tti, imgs_for_gen
            torch.cuda.empty_cache()

    texts_per_item: Dict[int, List[str]] = {i: [""] * K for i in range(len(items))}
    for (i, k), txt in zip(flat_meta, all_pred_texts):
        texts_per_item[i][k] = txt or ""

    results = []
    for i in range(len(items)):
        status = per_item_status[i]
        if status != "ok":
            results.append({
                "status": status,
                "candidates": [],
                "chosen_clean_text": None,
            })
            continue

        it = items[i]
        gt_letter = it['gt_letter']
        raw_texts = texts_per_item[i]
        clean_texts = [strip_noise_tags(t) for t in raw_texts]
        image_list = [it['image_pil']] * num_candidates
        if use_vds:
            try:
                vds_list = compute_vds_batch(embedder, image_list, clean_texts)
            except Exception as e:
                logger.warning(f"[vds] failed for qid={it.get('qid')}: {e}")
                vds_list = [0.0] * num_candidates
        else:
            vds_list = [0.0] * num_candidates

        cands = []
        for raw, clean, vds in zip(raw_texts, clean_texts, vds_list):
            pred_letter = normalize_letter(raw)
            acc_ok = (pred_letter == gt_letter) and (gt_letter != "")
            cands.append({
                "raw": raw, "clean": clean,
                "pred_letter": pred_letter,
                "vds": float(vds),
                "acc_ok": bool(acc_ok),
                "selected": False,
            })

        ok_cands = [c for c in cands if c["acc_ok"]]
        if not ok_cands:
            results.append({
                "status": "fallback_sft",
                "candidates": cands,
                "chosen_clean_text": None,
            })
            continue
        best = (max(ok_cands, key=lambda c: c["vds"]) if use_vds
                else random.choice(ok_cands))
        best["selected"] = True
        results.append({
            "status": "trained",
            "candidates": cands,
            "chosen_clean_text": best["clean"],
        })

    return results


def build_dataset(args, tokenizer, base, dtype):
    
    name = args.dataset_name.lower()

    if name == "omnimed":
        logger.info(">> Building OmniMedVQA_Dataset (with visdep/dep filters)")
        return OmniMedVQA_Dataset(
            root_dir=args.dataset_path,
            tokenizer=tokenizer,
            model=base,
            torch_type=dtype,
            input_length=args.max_input_len,
            output_length=args.max_output_len,
            noise_rate=0.0,
            use_doctor_prompt=args.use_doctor_prompt,
            visdep_sidecar=args.visdep_sidecar,
            visdep_min_score=args.visdep_min_score,
            visdep_weighting=args.visdep_weighting,
            dep_sidecar=args.dep_sidecar,
            dep_min=args.dep_min,
            keep_if_dep_missing=args.keep_if_dep_missing,
        )

    if name == "pmc_vqa":
        use_filter = args.pmc_use_filter
        logger.info(f">> Building PMC_VQA_Dataset (split={args.pmc_split}, "
                    f"use_filter={use_filter})")
        return PMC_VQA_Dataset(
            root_dir=args.dataset_path,
            tokenizer=tokenizer,
            model=base,
            torch_type=dtype,
            input_length=args.max_input_len,
            output_length=args.max_output_len,
            noise_rate=0.0,
            use_doctor_prompt=args.use_doctor_prompt,
            split=args.pmc_split,
            visdep_sidecar=args.visdep_sidecar if use_filter else None,
            visdep_min_score=args.visdep_min_score if use_filter else None,
            visdep_weighting=args.visdep_weighting,
            dep_sidecar=args.dep_sidecar if use_filter else None,
            dep_min=args.dep_min if use_filter else None,
            keep_if_dep_missing=True,
        )

    if name == "medxpertqa":
        use_filter = args.medx_use_filter
        logger.info(f">> Building MedXpertQA_Dataset "
                    f"(access={args.medx_access}, split_files={args.medx_split_files}, "
                    f"use_filter={use_filter})")
        return MedXpertQA_Dataset(
            root_dir=args.dataset_path,
            tokenizer=tokenizer,
            model=base,
            torch_type=dtype,
            input_length=args.max_input_len,
            output_length=args.max_output_len,
            noise_rate=0.0,
            use_doctor_prompt=args.use_doctor_prompt,
            access=args.medx_access,
            split_files=args.medx_split_files,
            visdep_sidecar=args.visdep_sidecar if use_filter else None,
            visdep_min_score=args.visdep_min_score if use_filter else None,
            visdep_weighting=args.visdep_weighting,
            dep_sidecar=args.dep_sidecar if use_filter else None,
            dep_min=args.dep_min if use_filter else None,
            keep_if_dep_missing=True,
        )

    raise ValueError(f"unknown --dataset_name: {args.dataset_name}")


def parse_args():
    p = argparse.ArgumentParser("Self-training with noise annotation")
    p.add_argument("--model_path", type=str,
                   default="checkpoints/sft")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Full training state checkpoint to resume from")
    p.add_argument("--dataset_path", type=str,
                   default="data/OmniMedVQA-Train")
    p.add_argument("--save_path", type=str,
                   default="checkpoints/sft")
    p.add_argument("--ds_config", type=str, default="finetune_demo/ds_config.yaml")
    p.add_argument("--torch_type", type=str, default="torch.bfloat16")

    p.add_argument("--dataset_name", type=str, default="omnimed",
                   choices=["omnimed", "pmc_vqa", "medxpertqa"],
                   help="training dataset")
    p.add_argument("--pmc_split", type=str, default="train",
                   help="PMC-VQA split")
    p.add_argument("--pmc_use_filter", action="store_true",
                   help="enable visdep/dep sidecar filtering for PMC-VQA")
    p.add_argument("--medx_access", type=str, default="MM",
                   help="MedXpertQA subset: MM or Text")
    p.add_argument("--medx_split_files", type=str, default="train",
                   help="MedXpertQA split keyword")
    p.add_argument("--medx_use_filter", action="store_true",
                   help="enable visdep/dep sidecar filtering for MedXpertQA")

    p.add_argument("--max_input_len", type=int, default=640)
    p.add_argument("--max_output_len", type=int, default=320,
                   help="token budget for long answers")
    p.add_argument("--train_dataset_rate", type=float, default=1.0)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4,
                   help="DataLoader batch size")

    p.add_argument("--num_candidates", type=int, default=4, help="number of sampled candidates K")
    p.add_argument("--use_vds", type=eval, default=True,
                   help="select the candidate with the highest VDS; if disabled, pick a random correct candidate")
    p.add_argument("--gen_top_p", type=float, default=0.95)
    p.add_argument("--gen_temperature", type=float, default=1.0)
    p.add_argument("--micro_batch_gen", type=int, default=4,
                   help="max rows per generate chunk")
    p.add_argument("--micro_batch_train", type=int, default=2,
                   help="samples per forward/backward pass"
                        "")

    p.add_argument("--curriculum", type=str, default="0,1,2,3,4,5",
                   help="comma-separated noise levels")

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

    p.add_argument("--lr", type=float, default=1e-6,
                   help="learning rate")
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--save_step", type=int, default=30,
                   help="save every N batches")

    p.add_argument("--visdep_sidecar", type=str,
                   default="data/OmniMedVQA/iodep_scores.jsonl")
    p.add_argument("--visdep_min_score", type=float, default=0.35)
    p.add_argument("--visdep_weighting", action="store_true")
    p.add_argument("--dep_sidecar", type=str,
                   default="data/OmniMedVQA/visdep_scores.jsonl")
    p.add_argument("--dep_min", type=float, default=0.0)
    p.add_argument("--keep_if_dep_missing", action="store_true")

    p.add_argument("--log_path", type=str,
                   default="logs")
    p.add_argument("--audit_jsonl", type=str, default=None,
                   help="per-sample JSONL audit log")
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    dtype = eval(args.torch_type)

    with open(args.ds_config, "r") as f:
        ds_cfg = yaml.safe_load(f)
    hf_ds_config = HfDeepSpeedConfig(ds_cfg)
    ds_plugin = DeepSpeedPlugin(hf_ds_config=hf_ds_config)
    accelerator = Accelerator(deepspeed_plugin=ds_plugin)
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True
    )
    if len(tokenizer) != base.get_input_embeddings().weight.size(0):
        base.resize_token_embeddings(len(tokenizer))

    dataset = build_dataset(args, tokenizer, base, dtype)

    qid_to_sample: Dict[str, Dict[str, Any]] = {}
    for s in dataset.samples:
        qid = s.get('question_id')
        if qid is not None:
            qid_to_sample[str(qid)] = s

    total = len(dataset)
    train_size = int(args.train_dataset_rate * total)
    val_size = total - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=dataset.custom_collate_fn,
    )

    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target, inference_mode=False,
        modules_to_save=["linear_proj"],
    )
    model = get_peft_model(base, lcfg)

    logger.info(">> Loading MedCLIP embedder ...")
    embedder = MedClipEmbedder()
    embedder.eval()
    logger.info(">> MedCLIP ready.")

    curriculum = [int(x) for x in args.curriculum.split(",") if x.strip()]
    assert all(0 <= x <= 5 for x in curriculum), f"invalid level in curriculum: {curriculum}"
    logger.info(f"Curriculum (per sample): {curriculum} (total {len(curriculum)} steps/sample)")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    per_sample_updates_upper = 2
    total_updates_est = len(train_loader) * args.num_epochs * per_sample_updates_upper
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optim, num_warmup_steps=args.warmup_steps,
        num_training_steps=total_updates_est,
    )
    model, optim, train_loader, lr_scheduler = accelerator.prepare(
        model, optim, train_loader, lr_scheduler
    )
    os.makedirs(args.save_path, exist_ok=True)

    batch_global = 0
    update_global = 0
    sample_global = 0
    resume_epoch = 0
    resume_step = 0

    if args.resume_from:
        accelerator.load_state(args.resume_from)
        resume_name = os.path.basename(args.resume_from.rstrip("/"))
        try:
            parts = resume_name.split("_")
            resume_epoch = int(parts[2])
            resume_step = int(parts[4])
            batch_global = resume_epoch * len(train_loader) + resume_step
        except Exception as e:
            logger.error(f"Failed to parse ckpt name '{resume_name}': {e}")
            raise
        logger.info(f"Resumed from {args.resume_from}: "
                    f"epoch={resume_epoch}, step={resume_step}, batch_global={batch_global}")

    audit_path = args.audit_jsonl or os.path.join(args.save_path, "audit.jsonl")
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
        writer = SummaryWriter(log_dir=args.save_path)
    else:
        writer = None

    n_total_seen = 0
    n_trained = 0
    n_skipped = 0
    skip_reasons: Dict[str, int] = {}

    def _bump_skip(reason: str):
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for epoch in range(args.num_epochs):
        if epoch < resume_epoch:
            continue

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}",
                    disable=not accelerator.is_main_process)
        for step, batch in enumerate(pbar):
            if epoch <= resume_epoch and step < resume_step:
                continue
            batch_global += 1
            B = len(batch.get("question_id", []))
            if B == 0:
                continue

            items: List[Optional[Dict[str, Any]]] = []
            for bi in range(B):
                qid = batch["question_id"][bi]
                qtype = batch.get("question_type", [""] * B)[bi]
                modality = batch.get("modality_type", [""] * B)[bi]
                question = batch.get("question", [None] * B)[bi] if "question" in batch else None
                image_path = batch.get("image_path", [None] * B)[bi] if "image_path" in batch else None
                raw_gt = batch.get("raw_gt_answer", [None] * B)[bi] if "raw_gt_answer" in batch else None

                matched = qid_to_sample.get(str(qid))
                if matched is None:
                    _bump_skip("missing_sample_lookup")
                    items.append(None)
                    continue

                try:
                    image_pil = Image.open(image_path or matched['image_path']).convert("RGB")
                except Exception as e:
                    _bump_skip("image_open_failed")
                    items.append(None)
                    continue

                options = matched['options']
                gt_with_letter = raw_gt or matched.get('gt_answer', '')
                for letter, opt in options:
                    if opt.strip().lower().rstrip(".") == (gt_with_letter or "").strip().lower().rstrip("."):
                        gt_with_letter = f"{letter}. {opt}"
                        break
                gt_letter = normalize_letter(gt_with_letter)

                items.append({
                    "qid": qid,
                    "question_type": qtype,
                    "modality_type": modality,
                    "question": question or matched['question'],
                    "image_pil": image_pil,
                    "image_path": image_path or matched['image_path'],
                    "options": options,
                    "raw_gt": raw_gt or matched.get('gt_answer', ''),
                    "gt_letter": gt_letter,
                    "gt_with_letter": gt_with_letter,
                })

            valid_items = [it for it in items if it is not None]
            n_total_seen += B
            if not valid_items:
                if accelerator.is_main_process:
                    with open(audit_path, "a", encoding="utf-8") as af:
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for _ in items:
                            n_skipped += 1
                            af.write(json.dumps({
                                "ts": ts, "batch_global": batch_global,
                                "epoch": epoch,
                                "status": "skipped_lookup_or_image",
                            }, ensure_ascii=False) + "\n")
                continue

            model.eval()
            t0 = time.time()
            filter_results = generate_and_filter_for_batch(
                model=accelerator.unwrap_model(model),
                tokenizer=tokenizer,
                dataset=dataset,
                embedder=embedder,
                items=valid_items,
                num_candidates=args.num_candidates,
                max_new_tokens=args.max_output_len,
                top_p=args.gen_top_p,
                temperature=args.gen_temperature,
                micro_batch_gen=args.micro_batch_gen,
                device=device,
                dtype=dtype,
                use_vds=args.use_vds,
            )
            model.train()
            gen_dt = time.time() - t0

            keep_items: List[Dict[str, Any]] = []
            keep_clean_texts: List[str] = []
            fallback_items: List[Dict[str, Any]] = []
            audit_entries: List[Dict[str, Any]] = []
            audit_idx_to_kind: List[str] = []

            for it, res in zip(valid_items, filter_results):
                sample_global += 1
                if res["status"] == "trained":
                    keep_items.append(it)
                    keep_clean_texts.append(res["chosen_clean_text"])
                    n_trained += 1
                    kind = "trained"
                elif res["status"] == "fallback_sft":
                    fallback_items.append(it)
                    n_trained += 1
                    kind = "fallback"
                else:
                    n_skipped += 1
                    _bump_skip(res["status"])
                    kind = "skipped"

                audit_entries.append({
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "batch_global": batch_global,
                    "sample_global": sample_global,
                    "epoch": epoch,
                    "qid": it["qid"],
                    "question_type": it["question_type"],
                    "modality_type": it["modality_type"],
                    "image_path": it["image_path"],
                    "gt_letter": it["gt_letter"],
                    "gt_text": it["raw_gt"],
                    "candidates": res["candidates"],
                    "status": "trained" if kind in ("trained", "fallback") else res["status"],
                    "fallback": (kind == "fallback"),
                    "chosen_clean_text": res["chosen_clean_text"]
                                          if kind == "trained"
                                          else (it.get("gt_with_letter") if kind == "fallback" else None),
                    "curriculum_mode": ("random_single" if kind == "trained" else "fallback_L0"),
                    "curriculum_levels": (curriculum if kind == "trained"
                                          else ([0] if kind == "fallback" else [])),
                    "losses_per_level_batch_mean": {},
                    "loss_fallback_batch_mean": None,
                })
                audit_idx_to_kind.append(kind)

            losses_table: Dict[str, List[float]] = {f"L{nl}": [] for nl in curriculum}
            if len(keep_items) > 0:
                idx_chunks = list(_chunks(list(range(len(keep_items))), args.micro_batch_train))
                for chunk in idx_chunks:
                    sample_dicts = []
                    chunk_levels = []
                    for j in chunk:
                        it = keep_items[j]
                        clean = keep_clean_texts[j]
                        nl = random.choice(curriculum)
                        chunk_levels.append(nl)
                        label_text = f"<noise>L{nl}</noise> " + clean.strip()
                        sd = build_single_sample(
                            dataset, tokenizer,
                            it['image_pil'], it['question'], it['options'],
                            label_text=label_text,
                            noise_level=nl,
                            device=device, dtype=dtype,
                        )
                        sample_dicts.append(sd)
                    mb = collate_samples(sample_dicts)

                    outputs = model(
                        input_ids=mb['input_ids'],
                        token_type_ids=mb['token_type_ids'],
                        attention_mask=mb['attention_mask'],
                        images=mb['images'],
                        labels=mb['labels'],
                        noise_level=mb['noise_level'],
                    )
                    loss = outputs.loss
                    optim.zero_grad(set_to_none=True)
                    accelerator.backward(loss)
                    optim.step()
                    lr_scheduler.step()

                    update_global += 1
                    _l = float(loss.detach().item())
                    for nl in set(chunk_levels):
                        losses_table[f"L{nl}"].append(_l)

                    if accelerator.is_main_process and writer is not None:
                        for nl in set(chunk_levels):
                            writer.add_scalar(f"loss_per_level/L{nl}", _l, update_global)
                        writer.add_scalar("loss/per_update", _l, update_global)

                    del outputs, loss, mb
                    torch.cuda.empty_cache()

            fallback_losses: List[float] = []
            if len(fallback_items) > 0:
                idx_chunks_fb = list(_chunks(list(range(len(fallback_items))), args.micro_batch_train))
                for chunk in idx_chunks_fb:
                    sample_dicts = []
                    for j in chunk:
                        it = fallback_items[j]
                        gt_with_letter = it.get('gt_with_letter') or it.get('raw_gt') or ""
                        label_text = "<noise>L0</noise> " + gt_with_letter.strip()
                        sd = build_single_sample_fallback(
                            dataset, tokenizer,
                            it['image_pil'], it['question'], it['options'],
                            label_text=label_text,
                            noise_level=0,
                            device=device, dtype=dtype,
                        )
                        sample_dicts.append(sd)
                    mb = collate_samples(sample_dicts)

                    outputs = model(
                        input_ids=mb['input_ids'],
                        token_type_ids=mb['token_type_ids'],
                        attention_mask=mb['attention_mask'],
                        images=mb['images'],
                        labels=mb['labels'],
                        noise_level=mb['noise_level'],
                    )
                    loss = outputs.loss
                    optim.zero_grad(set_to_none=True)
                    accelerator.backward(loss)
                    optim.step()
                    lr_scheduler.step()

                    update_global += 1
                    fallback_losses.append(float(loss.detach().item()))

                    if accelerator.is_main_process and writer is not None:
                        writer.add_scalar("loss_per_level/L0_fallback", loss.item(), update_global)
                        writer.add_scalar("loss/per_update", loss.item(), update_global)

                    del outputs, loss, mb
                    torch.cuda.empty_cache()

            if accelerator.is_main_process:
                shared_losses = {k: (sum(v) / len(v) if v else None)
                                 for k, v in losses_table.items()}
                fb_loss_mean = (sum(fallback_losses) / len(fallback_losses)
                                if fallback_losses else None)
                with open(audit_path, "a", encoding="utf-8") as af:
                    for entry, kind in zip(audit_entries, audit_idx_to_kind):
                        if kind == "trained":
                            entry["losses_per_level_batch_mean"] = shared_losses
                        elif kind == "fallback":
                            entry["loss_fallback_batch_mean"] = fb_loss_mean
                        af.write(json.dumps(entry, ensure_ascii=False) + "\n")

                with open(args.log_path, "a", encoding="utf-8") as lf:
                    ts = datetime.now().strftime("%m-%d %H:%M:%S")
                    parts_loss = []
                    if keep_items:
                        parts_loss.append(", ".join(
                            f"L{nl}={(shared_losses.get(f'L{nl}') or 0):.4f}" for nl in curriculum
                        ))
                    if fallback_items:
                        parts_loss.append(f"L0_fb={(fb_loss_mean or 0):.4f}")
                    loss_str = " | ".join(parts_loss) if parts_loss else "(no_train)"
                    lf.write(
                        f"[{ts}] ds={args.dataset_name} epoch={epoch} batch={batch_global} "
                        f"| seen={n_total_seen} train={n_trained} skip={n_skipped} "
                        f"| keep={len(keep_items)} fb={len(fallback_items)} / {len(valid_items)} "
                        f"| gen={gen_dt:.1f}s | updates={update_global} | {loss_str}\n"
                    )

                pbar.set_postfix({
                    "train": n_trained, "skip": n_skipped,
                    "keep": f"{len(keep_items)}/{len(valid_items)}",
                    "fb": len(fallback_items),
                    "upd": update_global,
                })
                if writer is not None:
                    writer.add_scalar("dist/keep_ratio_batch",
                                      len(keep_items) / max(len(valid_items), 1),
                                      batch_global)
                    writer.add_scalar("dist/fallback_ratio_batch",
                                      len(fallback_items) / max(len(valid_items), 1),
                                      batch_global)
                    writer.add_scalar("dist/cum_trained", n_trained, batch_global)
                    writer.add_scalar("dist/cum_skipped", n_skipped, batch_global)
                    writer.add_scalar("perf/gen_seconds_per_batch", gen_dt, batch_global)

            for it in valid_items:
                try:
                    it['image_pil'].close()
                except Exception:
                    pass

            if batch_global % args.save_step == 0 and batch_global > 0:
                ckpt = os.path.join(args.save_path, f"checkpoint_epoch_{epoch}_step_{step + 1}")
                os.makedirs(ckpt, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(ckpt, safe_serialization=True)
                accelerator.save_state(ckpt)
                if accelerator.is_main_process:
                    logger.info(f"[ckpt] saved {ckpt} (updates={update_global}, "
                                f"train={n_trained}, skip={n_skipped})")

            gc.collect()
            torch.cuda.empty_cache()

    final_ckpt = os.path.join(args.save_path, "final_model")
    os.makedirs(final_ckpt, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(final_ckpt, safe_serialization=True)
    accelerator.save_state(final_ckpt)

    if accelerator.is_main_process:
        logger.info(
            f"Training finished. ds={args.dataset_name} "
            f"seen={n_total_seen} train={n_trained} "
            f"skip={n_skipped} updates={update_global} "
            f"skip_reasons={skip_reasons}"
        )


if __name__ == "__main__":
    main()
