# -*- coding: utf-8 -*-
"""
Exp P0-1: Paired vs. Mismatched / Within-Modality / Counterfactual VDS
Evaluates already-saved Sub2000 responses from eval_vds_cogvlm and eval_noise_lingshu
using BMCA, PLIP, and UniMed-CLIP on GPU 0.
"""
import os
import sys
import json
import math
import random
import glob
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F

sys.path.insert(0, "/path/to/NAP_code")
from rescore_vds import (
    OpenCLIPEmbedder,
    TransformersCLIPEmbedder,
    UniMedCLIPEmbedder,
    get_clean_text,
    make_counterfactual,
)

DEVICE = "cuda:0"
OUT_DIR = "/path/to/revision_outputs/exp1_mismatched_vds"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load Sub2000 metadata (qid -> image_path, modality_type, gt_answer)
SUB2000_ROOT = "/path/to/datasets/OmniMedVQA-Clean-Sub2000"
qa_dir = os.path.join(SUB2000_ROOT, "QA_information", "Open-access")
meta = {}
mod_to_qids = defaultdict(list)
for fn in sorted(os.listdir(qa_dir)):
    if not fn.endswith(".json"):
        continue
    for it in json.load(open(os.path.join(qa_dir, fn), encoding="utf-8")):
        qid = str(it["question_id"])
        ip = it["image_path"]
        cand = ip if os.path.isabs(ip) else os.path.join(SUB2000_ROOT, ip)
        if not os.path.exists(cand):
            cand = os.path.join(SUB2000_ROOT, "Images", ip)
        meta[qid] = {
            "img_path": cand,
            "mod": it.get("modality_type", ""),
            "task": it.get("question_type", ""),
            "gt_answer": str(it.get("gt_answer", "")).strip().lower(),
        }
        mod_to_qids[meta[qid]["mod"]].append(qid)

qids = sorted(meta.keys())
print(f"[Exp1] Loaded {len(qids)} Sub2000 items across {len(mod_to_qids)} modalities")

# Build deterministic mismatch mappings (seed 42):
# (a) random mismatch (different image_path)
# (b) within-modality clinically different mismatch (same modality, different image_path AND different gt_answer)
rng = random.Random(42)
rand_map = {}
same_mod_diff_map = {}

for q in qids:
    m = meta[q]
    # random mismatch
    for _ in range(100):
        cand_q = rng.choice(qids)
        if meta[cand_q]["img_path"] != m["img_path"]:
            rand_map[q] = cand_q
            break
    # within-modality clinically different mismatch
    pool = mod_to_qids[m["mod"]]
    diff_pool = [cq for cq in pool if meta[cq]["img_path"] != m["img_path"] and meta[cq]["gt_answer"] != m["gt_answer"]]
    if not diff_pool:
        diff_pool = [cq for cq in pool if meta[cq]["img_path"] != m["img_path"]]
    same_mod_diff_map[q] = rng.choice(diff_pool)

# Pre-load unique images (1998 images) and their counterfactual blurred+shuffled versions
unique_paths = sorted(set(meta[q]["img_path"] for q in qids))
path_to_idx = {p: i for i, p in enumerate(unique_paths)}
print(f"[Exp1] Loading {len(unique_paths)} unique images...")
pil_real = [Image.open(p).convert("RGB") for p in unique_paths]
pil_cf = []
for im in pil_real:
    b = im.filter(ImageFilter.GaussianBlur(radius=3.0))
    c = make_counterfactual(b, mode="shuffle", blur_sigma=3.0, shuffle_grid=4)
    pil_cf.append(c)

# Configurations from user's approved paths ONLY
vds_root = "/path/to/NAP_code/eval/vlcs-plus-curriculum/eval_vds_cogvlm"
ling_root = "/path/to/NAP_code/eval/qwen/eval_noise_lingshu"

cfgs = {
    "sft": f"{vds_root}/eval_vlcs_sft/vds_predictions_bmca_clean.jsonl",
    "no_noise": glob.glob(f"{vds_root}/eval_vlcs_no_noise/**/vds_predictions_bmca_clean.jsonl", recursive=True)[0],
    "no_np": f"{vds_root}/eval_vlcs_no_np-300/vds_predictions_bmca_clean.jsonl",
    "no_vec": f"{vds_root}/eval_vlcs_no_vec/vds_predictions_bmca_clean.jsonl",
    "no_visonlora": f"{vds_root}/eval_vlcs_no_visonlora/vds_predictions_bmca_clean.jsonl",
    "full": f"{vds_root}/eval_vlcs_full/vds_predictions_bmca_clean.jsonl",
    "lingshu_base": f"{ling_root}/lingshu-7B/vds_noise_predictions_bmca_clean.jsonl",
    "lingshu_nap": f"{ling_root}/lingshu-7B-full/vds_noise_predictions_bmca_clean.jsonl",
}

cfg_texts = {}
for cname, path in cfgs.items():
    rows = {str(json.loads(l)["question_id"]): json.loads(l) for l in open(path, encoding="utf-8") if l.strip()}
    cfg_texts[cname] = [get_clean_text(rows[q]) for q in qids]

def encode_images_batched(emb, imgs, bs=64):
    feats = []
    for i in range(0, len(imgs), bs):
        f = emb.encode_image(imgs[i:i+bs])
        feats.append(F.normalize(f, dim=-1).cpu())
    return torch.cat(feats, dim=0)

def encode_texts_batched(emb, texts, bs=128):
    feats = []
    for i in range(0, len(texts), bs):
        f = emb.encode_text(texts[i:i+bs])
        feats.append(F.normalize(f, dim=-1).cpu())
    return torch.cat(feats, dim=0)

def cluster_ci95_arr(vals):
    img_to_v = defaultdict(list)
    for q, v in zip(qids, vals):
        img_to_v[meta[q]["img_path"]].append(float(v))
    N = len(vals)
    G = len(img_to_v)
    mu = float(np.mean(vals))
    ss = sum((sum(vs) - mu * len(vs)) ** 2 for vs in img_to_v.values())
    se = math.sqrt((G / (G - 1.0)) * ss / (N * N))
    return [mu - 1.96 * se, mu + 1.96 * se]

idx_real = torch.tensor([path_to_idx[meta[q]["img_path"]] for q in qids], dtype=torch.long)
idx_rand = torch.tensor([path_to_idx[meta[rand_map[q]]["img_path"]] for q in qids], dtype=torch.long)
idx_smod = torch.tensor([path_to_idx[meta[same_mod_diff_map[q]]["img_path"]] for q in qids], dtype=torch.long)

evaluators = [
    ("bmca", lambda: OpenCLIPEmbedder(hf_hub="", device=DEVICE, local_pt_path="/path/to/clip_models/BMC_CLIP_CF.pt")),
    ("plip", lambda: TransformersCLIPEmbedder(model_path="/path/to/clip_models/PLIP", device=DEVICE)),
    ("unimed", lambda: UniMedCLIPEmbedder(
        local_pt_path="/path/to/clip_models/unimed_clip_vit_l14_base_text_encoder.pt",
        tokenizer_path="/path/to/clip_models/BiomedBERT_tok",
        device=DEVICE,
    )),
]

results = {}
for ev_name, builder in evaluators:
    print(f"\n[Exp1] Running evaluator: {ev_name}")
    emb = builder()
    img_real_f = encode_images_batched(emb, pil_real, bs=64)
    img_cf_f = encode_images_batched(emb, pil_cf, bs=64)

    f_real = img_real_f[idx_real]
    f_rand = img_real_f[idx_rand]
    f_smod = img_real_f[idx_smod]
    f_cf = img_cf_f[idx_real]

    ev_res = {}
    for cname, texts in cfg_texts.items():
        t_f = encode_texts_batched(emb, texts, bs=128)
        s_real = (f_real * t_f).sum(-1).numpy()
        s_rand = (f_rand * t_f).sum(-1).numpy()
        s_smod = (f_smod * t_f).sum(-1).numpy()
        s_cf = (f_cf * t_f).sum(-1).numpy()

        d_rand = s_real - s_rand
        d_smod = s_real - s_smod
        d_cf = s_real - s_cf

        ev_res[cname] = {
            "paired_mean": float(s_real.mean()),
            "rand_mismatch_mean": float(s_rand.mean()),
            "same_mod_mismatch_mean": float(s_smod.mean()),
            "cf_degraded_mean": float(s_cf.mean()),
            "gap_rand_mean": float(d_rand.mean()),
            "gap_rand_ci95": cluster_ci95_arr(d_rand),
            "gap_same_mod_mean": float(d_smod.mean()),
            "gap_same_mod_ci95": cluster_ci95_arr(d_smod),
            "gap_cf_mean": float(d_cf.mean()),
            "gap_cf_ci95": cluster_ci95_arr(d_cf),
        }
        print(
            f"  {cname:13s} | paired={s_real.mean():.4f} | same_mod={s_smod.mean():.4f} (gap={d_smod.mean():+.4f}) "
            f"| rand={s_rand.mean():.4f} (gap={d_rand.mean():+.4f}) | cf={s_cf.mean():.4f} (gap={d_cf.mean():+.4f})"
        )
    results[ev_name] = ev_res
    del emb
    torch.cuda.empty_cache()

out_path = os.path.join(OUT_DIR, "mismatched_vds_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[Exp1] DONE -> {out_path}")
