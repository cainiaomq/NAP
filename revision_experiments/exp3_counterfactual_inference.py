# -*- coding: utf-8 -*-
"""
Exp P0-3: Direct Image-Dependency Counterfactual Inference Test (Reviewer #3 Comment 6 & 12)
Evaluates CogVLM2-SFT, w/o ST + R_np, and NAP-19B on a stratified 400-question subset of Sub2000
(50 per modality, seed=42) under 4 visual input conditions:
  (1) real       : true paired medical image
  (2) blank      : uniform black/zero image (no-image text-prior baseline)
  (3) shuffled   : randomly mismatched image from another question
  (4) same_mod   : visually similar but clinically different image (same modality, different gt_answer)
"""
import os
import re
import sys
import json
import math
import random
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/path/to/NAP_code")
from utils.omnimedqkv import OmniMedVQA_Dataset

DEVICE = "cuda:0"
OUT_DIR = "/path/to/revision_outputs/exp3_counterfactual_infer"
os.makedirs(OUT_DIR, exist_ok=True)

SUB2000_ROOT = "/path/to/datasets/OmniMedVQA-Clean-Sub2000"
qa_dir = os.path.join(SUB2000_ROOT, "QA_information", "Open-access")

all_items = []
by_mod = defaultdict(list)
for fn in sorted(os.listdir(qa_dir)):
    if not fn.endswith(".json"):
        continue
    for it in json.load(open(os.path.join(qa_dir, fn), encoding="utf-8")):
        ip = it["image_path"]
        cand = ip if os.path.isabs(ip) else os.path.join(SUB2000_ROOT, ip)
        if not os.path.exists(cand):
            cand = os.path.join(SUB2000_ROOT, "Images", ip)
        opts = [(k[-1], it.get(k)) for k in ["option_A", "option_B", "option_C", "option_D"] if it.get(k)]
        gt_raw = str(it.get("gt_answer", "")).strip()
        gt_letter = ""
        for ltr, txt in opts:
            if str(txt).strip().lower() == gt_raw.lower():
                gt_letter = ltr
                break
        rec = {
            "question_id": str(it["question_id"]),
            "image_path": cand,
            "modality_type": it.get("modality_type", ""),
            "question_type": it.get("question_type", ""),
            "question": it.get("question", ""),
            "options": opts,
            "gt_answer": gt_raw,
            "gt_letter": gt_letter,
        }
        all_items.append(rec)
        by_mod[rec["modality_type"]].append(rec)

# Stratified 240-question sample (30 per modality across all 8 modalities, seed=42)
rng = random.Random(42)
selected = []
for mod in sorted(by_mod.keys()):
    pool = list(by_mod[mod])
    rng.shuffle(pool)
    selected.extend(pool[:30])

print(f"[Exp3] Selected {len(selected)} questions (30/modality x {len(by_mod)} modalities)")

# Pre-build deterministic mismatch paths for each selected item
for it in selected:
    # shuffled (random from all_items with different image_path)
    while True:
        c = rng.choice(all_items)
        if c["image_path"] != it["image_path"]:
            it["shuffled_path"] = c["image_path"]
            break
    # same_mod (same modality, different image_path AND different gt_answer)
    mpool = [c for c in by_mod[it["modality_type"]] if c["image_path"] != it["image_path"] and c["gt_answer"].lower() != it["gt_answer"].lower()]
    if not mpool:
        mpool = [c for c in by_mod[it["modality_type"]] if c["image_path"] != it["image_path"]]
    it["same_mod_path"] = rng.choice(mpool)["image_path"]

_NOISE_TAG_RE = re.compile(r"<noise>.*?</noise>\s*", re.I | re.S)
_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Da-d1-4])")
_NUM2L = {"1": "A", "2": "B", "3": "C", "4": "D"}

def extract_letter(txt):
    t = _NOISE_TAG_RE.sub("", txt or "").strip()
    m = _LETTER_RE.search(t)
    if not m:
        return ""
    c = m.group(1).upper()
    return _NUM2L.get(c, c if c in "ABCD" else "")

DOCTOR_P1 = (
    "You are a medical doctor analyzing a clinical image. "
    "The image may have been corrupted by Gaussian noise injected into its visual embeddings, "
    "with severity ranging from L0 (clean) to L5 (severe degradation):\n"
    "  L0 = clean, all anatomical details clearly visible\n"
    "  L1 = barely perceptible noise, diagnosis unaffected\n"
    "  L2 = mild noise, fine textures slightly blurred\n"
    "  L3 = moderate noise, some structures harder to distinguish\n"
    "  L4 = strong noise, key features partially obscured\n"
    "  L5 = severe noise, most visual evidence unreliable\n"
    "First silently assess the noise level, then answer the question "
    "based ONLY on the visual evidence you can still reliably perceive."
)
DOCTOR_P2 = (
    "You MUST respond in EXACTLY this format on a single line, with no extra text:\n"
    "<noise>L<digit></noise> <letter>. <option text>: Evidence: <one concise sentence>\n\n"
    "Where:\n"
    "  - <digit> is one integer from 0 to 5 (your noise estimate). e.g., <noise>L3</noise>.\n"
    "  - <letter> is one of A, B, C, D (your chosen option)\n"
    "  - <option text> is the exact text of the chosen option\n"
    "  - Evidence is ONE concise sentence describing the key visual cue supporting your choice\n\n"
    "Output the answer line now:"
)

def build_query(it):
    opt_text = "\n".join(f"{l}. {t}" for l, t in it["options"])
    return DOCTOR_P1 + f"\n\nQuestion:\n{it['question']}\nOptions:\n{opt_text}\nAnswer:\n" + DOCTOR_P2

MODELS = [
    ("NAP-19B", "/path/to/models/THUDM/vlcs2/cogvlm2-llama3-chat-vlcs-19B-sft-medreason2026-vdrfv2-400"),
    ("w/o_ST+R_np", "/path/to/models/THUDM/vlcs/plus/cogvlm2-llama3-chat-vlcs-19B-vdrfv2_non-2000"),
    ("CogVLM2-SFT", "/path/to/models/THUDM/vlcs/cogvlm2-llama3-chat-vlcs-19B-sft-plus"),
]

CONDITIONS = ["real", "blank", "shuffled", "same_mod"]
blank_img = Image.new("RGB", (512, 512), (0, 0, 0))

summary_all = {}

for mname, mpath in MODELS:
    print(f"\n[Exp3] === Model: {mname} ===")
    tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mpath, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval().to(DEVICE)

    m_summary = {}
    real_preds = {}
    out_jsonl = os.path.join(OUT_DIR, f"preds_{mname.replace('/', '_')}.jsonl")
    fout = open(out_jsonl, "w", encoding="utf-8")

    for cond in CONDITIONS:
        correct = 0
        flipped = 0
        total = len(selected)
        for idx, it in enumerate(selected):
            if cond == "real":
                im = Image.open(it["image_path"]).convert("RGB")
            elif cond == "blank":
                im = blank_img
            elif cond == "shuffled":
                im = Image.open(it["shuffled_path"]).convert("RGB")
            else:
                im = Image.open(it["same_mod_path"]).convert("RGB")

            query = build_query(it)
            inp = model.build_conversation_input_ids(
                tokenizer=tokenizer, query=query, history=None, images=[im]
            )
            gen_in = {
                "input_ids": inp["input_ids"].unsqueeze(0).to(DEVICE),
                "token_type_ids": inp["token_type_ids"].unsqueeze(0).to(DEVICE),
                "attention_mask": inp["attention_mask"].unsqueeze(0).to(DEVICE),
                "images": [[inp["images"][0].to(DEVICE, dtype=torch.bfloat16)]],
            }
            with torch.no_grad():
                out = model.generate(
                    **gen_in,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or 128002,
                )
            P = gen_in["input_ids"].shape[1]
            pred_text = tokenizer.decode(out[0, P:], skip_special_tokens=True).strip()
            pl = extract_letter(pred_text)
            ok = (pl == it["gt_letter"])
            if ok:
                correct += 1
            if cond == "real":
                real_preds[it["question_id"]] = pl
            else:
                if pl != real_preds.get(it["question_id"], ""):
                    flipped += 1
            fout.write(json.dumps({
                "model": mname,
                "condition": cond,
                "question_id": it["question_id"],
                "modality_type": it["modality_type"],
                "question_type": it["question_type"],
                "gt_letter": it["gt_letter"],
                "pred_letter": pl,
                "correct": ok,
                "pred_text": pred_text,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            if (idx + 1) % 60 == 0:
                print(f"  [{mname} | {cond}] {idx+1}/{total} acc={correct/(idx+1)*100:.2f}%", flush=True)

        acc = correct / total
        se = math.sqrt(acc * (1 - acc) / total)
        m_summary[cond] = {
            "total": total,
            "correct": correct,
            "acc": acc,
            "ci95": [acc - 1.96 * se, acc + 1.96 * se],
            "flip_rate_vs_real": (flipped / total) if cond != "real" else 0.0,
        }
        print(f"  >>> {mname} | {cond:10s} | Acc={acc*100:.2f}% [{(acc-1.96*se)*100:.2f}, {(acc+1.96*se)*100:.2f}] | Flip={m_summary[cond]['flip_rate_vs_real']*100:.2f}%", flush=True)

    fout.close()
    summary_all[mname] = m_summary
    del model, tokenizer
    torch.cuda.empty_cache()

with open(os.path.join(OUT_DIR, "counterfactual_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary_all, f, indent=2, ensure_ascii=False)
print("\n[Exp3] DONE!")
