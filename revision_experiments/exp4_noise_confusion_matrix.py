# -*- coding: utf-8 -*-
"""
Exp P0-4: Balanced L0-L5 Noise-Level Prediction Confusion Matrix & 95% CIs (Reviewer #3 Comment 11)
Runs BOTH NAP-19B (deep fusion, --arch cogvlm) and Lingshu-7B (+ST+RL, --arch lingshu)
on the same balanced 1,200-question stratified subset of Sub2000 (200 per level L0-L5).
"""
import os
import re
import sys
import json
import math
import random
import argparse
from collections import defaultdict, Counter

from PIL import Image
import torch

OUT_DIR = "/path/to/revision_outputs/exp4_noise_cm"
os.makedirs(OUT_DIR, exist_ok=True)

SUB2000_ROOT = "/path/to/datasets/OmniMedVQA-Clean-Sub2000"
qa_dir = os.path.join(SUB2000_ROOT, "QA_information", "Open-access")

LEVEL_RATIO = {0: 0.0, 1: 0.072, 2: 0.144, 3: 0.216, 4: 0.288, 5: 0.360}
_NOISE_LVL_RE = re.compile(r"<noise>\s*[Ll]?\s*(\d+)\s*</noise>", re.I)
_NOISE_TAG_RE = re.compile(r"<noise>.*?</noise>\s*", re.I | re.S)
_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Da-d1-4])")
_NUM2L = {"1": "A", "2": "B", "3": "C", "4": "D"}

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

def load_balanced_1200():
    all_items = []
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
            all_items.append({
                "question_id": str(it["question_id"]),
                "image_path": cand,
                "modality_type": it.get("modality_type", ""),
                "question_type": it.get("question_type", ""),
                "question": it.get("question", ""),
                "options": opts,
                "gt_letter": gt_letter,
            })
    rng = random.Random(42)
    rng.shuffle(all_items)
    sub = all_items[:1200]
    # Assign exactly 200 samples per level L0..L5 (shuffled deterministically)
    levels = [l for l in range(6) for _ in range(200)]
    rng.shuffle(levels)
    for it, lv in zip(sub, levels):
        it["true_level"] = lv
    return sub

def parse_level(txt):
    m = _NOISE_LVL_RE.search(txt or "")
    if not m:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 5 else None

def parse_letter(txt):
    t = _NOISE_TAG_RE.sub("", txt or "").strip()
    m = _LETTER_RE.search(t)
    if not m:
        return ""
    c = m.group(1).upper()
    return _NUM2L.get(c, c if c in "ABCD" else "")

def wilson_ci(k, n):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return [max(0.0, center - half), min(1.0, center + half)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["cogvlm", "lingshu"], required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    items = load_balanced_1200()
    device = "cuda:0"
    print(f"[Exp4] {args.tag} ({args.arch}) on {len(items)} balanced L0-L5 items")

    if args.arch == "cogvlm":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).eval().to(device)
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16
        ).eval().to(device)
        cur_level = [0]
        def _hook(mod, inp, out):
            if cur_level[0] > 0:
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    t = out.pooler_output
                elif isinstance(out, tuple):
                    t = out[0]
                else:
                    t = out
                ratio = LEVEL_RATIO[cur_level[0]]
                t.add_(torch.randn_like(t) * (ratio * t.float().std()).to(t.dtype))
            return out
        vis_mod = getattr(model, "visual", None) or model.model.visual
        vis_mod.register_forward_hook(_hook)

    cm = {f"L{i}": {f"L{j}": 0 for j in range(6)} | {"MISS": 0} for i in range(6)}
    vqa_by_lv = {i: [0, 0] for i in range(6)}
    out_jsonl = os.path.join(OUT_DIR, f"preds_{args.tag}.jsonl")

    torch.manual_seed(42)
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for idx, it in enumerate(items):
            im = Image.open(it["image_path"]).convert("RGB")
            lv = it["true_level"]
            q = build_query(it)
            with torch.no_grad():
                if args.arch == "cogvlm":
                    inp = model.build_conversation_input_ids(tokenizer=tokenizer, query=q, history=None, images=[im])
                    gen_in = {
                        "input_ids": inp["input_ids"].unsqueeze(0).to(device),
                        "token_type_ids": inp["token_type_ids"].unsqueeze(0).to(device),
                        "attention_mask": inp["attention_mask"].unsqueeze(0).to(device),
                        "images": [[inp["images"][0].to(device, dtype=torch.bfloat16)]],
                        "noise_level": [lv],
                    }
                    out = model.generate(**gen_in, max_new_tokens=32, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id or 128002)
                    pred_text = tokenizer.decode(out[0, gen_in["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                else:
                    cur_level[0] = lv
                    msg = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": q}]}]
                    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    ii, vi = process_vision_info(msg)
                    inputs = processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(device)
                    out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
                    pred_text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

            plv = parse_level(pred_text)
            pl = parse_letter(pred_text)
            vqa_ok = (pl == it["gt_letter"])
            cm[f"L{lv}"][f"L{plv}" if plv is not None else "MISS"] += 1
            vqa_by_lv[lv][1] += 1
            if vqa_ok:
                vqa_by_lv[lv][0] += 1
            fout.write(json.dumps({
                "question_id": it["question_id"],
                "true_level": lv,
                "pred_level": plv,
                "gt_letter": it["gt_letter"],
                "pred_letter": pl,
                "vqa_correct": vqa_ok,
                "pred_text": pred_text,
            }, ensure_ascii=False) + "\n")
            if (idx + 1) % 200 == 0:
                tot_c = sum(cm[f"L{i}"][f"L{i}"] for i in range(6))
                print(f"  [{args.tag}] {idx+1}/{len(items)} L0-5 exact={tot_c/(idx+1)*100:.2f}%", flush=True)

    # Compute summary
    tot_all = len(items)
    cor_all = sum(cm[f"L{i}"][f"L{i}"] for i in range(6))
    within1_all = sum(cm[f"L{i}"][f"L{j}"] for i in range(6) for j in range(6) if abs(i - j) <= 1)
    tot_l15 = sum(sum(cm[f"L{i}"].values()) for i in range(1, 6))
    cor_l15 = sum(cm[f"L{i}"][f"L{i}"] for i in range(1, 6))
    within1_l15 = sum(cm[f"L{i}"][f"L{j}"] for i in range(1, 6) for j in range(6) if abs(i - j) <= 1)

    per_level = {}
    for i in range(6):
        n_i = sum(cm[f"L{i}"].values())
        c_i = cm[f"L{i}"][f"L{i}"]
        vc, vn = vqa_by_lv[i]
        per_level[f"L{i}"] = {
            "n": n_i,
            "noise_exact": c_i / n_i,
            "noise_exact_ci95": wilson_ci(c_i, n_i),
            "vqa_acc": vc / vn,
            "vqa_acc_ci95": wilson_ci(vc, vn),
        }

    summary = {
        "tag": args.tag,
        "n_total": tot_all,
        "L0_L5_exact": cor_all / tot_all,
        "L0_L5_exact_ci95": wilson_ci(cor_all, tot_all),
        "L0_L5_within1": within1_all / tot_all,
        "L1_L5_exact": cor_l15 / tot_l15,
        "L1_L5_exact_ci95": wilson_ci(cor_l15, tot_l15),
        "L1_L5_within1": within1_l15 / tot_l15,
        "per_level": per_level,
        "confusion_matrix": cm,
    }
    out_json = os.path.join(OUT_DIR, f"summary_{args.tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Exp4] {args.tag} SUMMARY: L0-5 exact={summary['L0_L5_exact']*100:.2f}% (within±1={summary['L0_L5_within1']*100:.2f}%), L1-5 exact={summary['L1_L5_exact']*100:.2f}% -> {out_json}")

if __name__ == "__main__":
    main()
