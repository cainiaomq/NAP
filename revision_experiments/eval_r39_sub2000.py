# -*- coding: utf-8 -*-
"""
In-memory merge & full Sub2000 evaluation (Acc + BMCA/PLIP/UniMed VDS + Mismatched VDS)
for R3.9 300-step GRPO ablation checkpoints.
"""
import os, re, sys, json, math, shutil, random, tempfile, argparse
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file, save_file

sys.path.insert(0, "/path/to/NAP_code")
from rescore_vds import OpenCLIPEmbedder, TransformersCLIPEmbedder, UniMedCLIPEmbedder

SUB2000_ROOT = "/path/to/datasets/OmniMedVQA-Clean-Sub2000"
BMCA_PATH = "/path/to/clip_models/BMC_CLIP_CF.pt"
PLIP_PATH = "/path/to/clip_models/PLIP"
UNIMED_PATH = "/path/to/clip_models/unimed_clip_vit_l14_base_text_encoder.pt"
UNIMED_TOK = "/path/to/clip_models/BiomedBERT_tok"

_NOISE_TAG_RE = re.compile(r"<noise>.*?</noise>\s*", re.I | re.S)
_NOISE_LVL_RE = re.compile(r"<noise>\s*[Ll]?\s*(\d+)\s*</noise>", re.I)
_LETTER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Da-d1-4])")
_NUM2L = {"1": "A", "2": "B", "3": "C", "4": "D"}

NOISE_P1 = (
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
NOISE_P2 = (
    "You MUST respond in EXACTLY this format on a single line, with no extra text:\n"
    "<noise>L<digit></noise> <letter>. <option text>: Evidence: <one concise sentence>\n\n"
    "Where:\n"
    "  - <digit> is one integer from 0 to 5 (your noise estimate). e.g., <noise>L3</noise>.\n"
    "  - <letter> is one of A, B, C, D (your chosen option)\n"
    "  - <option text> is the exact text of the chosen option\n"
    "  - Evidence is ONE concise sentence describing the key visual cue supporting your choice\n\n"
    "Output the answer line now:"
)
PASSIVE_P1 = (
    "You are a medical doctor analyzing a clinical image. "
    "Base your reasoning strictly on the visible evidence in the image."
)
PASSIVE_P2 = (
    "You MUST respond in EXACTLY this format on a single line, with no extra text:\n"
    "<letter>. <option text>: Evidence: <one concise sentence>\n\n"
    "Where:\n"
    "  - <letter> is one of A, B, C, D (your chosen option)\n"
    "  - <option text> is the exact text of the chosen option\n"
    "  - Evidence is ONE concise sentence describing the key visual cue supporting your choice\n\n"
    "Output the answer line now:"
)


def extract_letter(txt):
    t = _NOISE_TAG_RE.sub("", txt or "").strip()
    m = _LETTER_RE.search(t)
    if not m:
        return ""
    c = m.group(1).upper()
    return _NUM2L.get(c, c if c in "ABCD" else "")


def strip_noise(txt):
    return _NOISE_TAG_RE.sub("", txt or "").strip()


def load_merged_model(base_path, adapter_path, device):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    if adapter_path:
        sd_path = os.path.join(adapter_path, "adapter_model.safetensors")
        sd = load_file(sd_path) if os.path.exists(sd_path) else torch.load(
            os.path.join(adapter_path, "adapter_model.bin"), map_location="cpu"
        )
        lora_sd, proj_sd = {}, {}
        for k, v in sd.items():
            if "linear_proj" in k:
                new_k = k.replace("base_model.model.", "").replace("linear_proj.linear_proj", "linear_proj")
                proj_sd[new_k] = v
            else:
                lora_sd[k] = v
        tmp_dir = tempfile.mkdtemp()
        shutil.copy(os.path.join(adapter_path, "adapter_config.json"), tmp_dir)
        with open(os.path.join(tmp_dir, "adapter_config.json"), "r") as f:
            cfg = json.load(f)
        cfg["modules_to_save"] = None
        with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
            json.dump(cfg, f)
        save_file(lora_sd, os.path.join(tmp_dir, "adapter_model.safetensors"))
        model = PeftModel.from_pretrained(model, tmp_dir)
        model = model.merge_and_unload()
        missing, unexpected = model.load_state_dict(proj_sd, strict=False)
        shutil.rmtree(tmp_dir)
        print(f"[Merge] In-memory merged {adapter_path} (proj keys={len(proj_sd)}, missing={len(missing)}, unexpected={len(unexpected)})")
    return model.eval().to(device), tokenizer


def cluster_ci95(items, vals):
    by_img = defaultdict(list)
    for it, v in zip(items, vals):
        by_img[it["image_path"]].append(v)
    N = len(vals)
    G = len(by_img)
    mu = sum(vals) / N
    ss = sum((sum(vs) - mu * len(vs)) ** 2 for vs in by_img.values())
    se = math.sqrt(max(0.0, (G / max(G - 1, 1)) * ss / (N * N)))
    return [mu - 1.96 * se, mu + 1.96 * se]


@torch.no_grad()
def score_3clip(items, texts, device):
    pils = [Image.open(it["image_path"]).convert("RGB") for it in items]
    rng = random.Random(42)
    by_mod = defaultdict(list)
    for i, it in enumerate(items):
        by_mod[it["modality_type"]].append(i)
    same_mod_idx = []
    rand_idx = []
    for i, it in enumerate(items):
        mp = [j for j in by_mod[it["modality_type"]] if items[j]["image_path"] != it["image_path"]]
        same_mod_idx.append(rng.choice(mp))
        while True:
            j = rng.randrange(len(items))
            if items[j]["image_path"] != it["image_path"]:
                rand_idx.append(j)
                break

    evaluators = [
        ("bmca", lambda: OpenCLIPEmbedder(hf_hub="", device=device, local_pt_path=BMCA_PATH)),
        ("plip", lambda: TransformersCLIPEmbedder(model_path=PLIP_PATH, device=device)),
        ("unimed", lambda: UniMedCLIPEmbedder(local_pt_path=UNIMED_PATH, tokenizer_path=UNIMED_TOK, device=device)),
    ]

    out = {}
    for ev_name, builder in evaluators:
        emb = builder()
        img_embs, txt_embs = [], []
        for i in range(0, len(items), 64):
            img_embs.append(F.normalize(emb.encode_image(pils[i:i+64]).float(), dim=-1).cpu())
            txt_embs.append(F.normalize(emb.encode_text(texts[i:i+64]).float(), dim=-1).cpu())
        IE = torch.cat(img_embs, 0)
        TE = torch.cat(txt_embs, 0)
        paired = (IE * TE).sum(-1).numpy()
        same_m = (IE[same_mod_idx] * TE).sum(-1).numpy()
        rand_m = (IE[rand_idx] * TE).sum(-1).numpy()
        out[ev_name] = {
            "paired": float(paired.mean()),
            "same_mod": float(same_m.mean()),
            "gap_same_mod": float((paired - same_m).mean()),
            "rand": float(rand_m.mean()),
            "gap_rand": float((paired - rand_m).mean()),
        }
        del emb
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--prompt_mode", choices=["noise", "passive"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_items", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    qa_dir = os.path.join(SUB2000_ROOT, "QA_information", "Open-access")
    items = []
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
            items.append({
                "question_id": str(it["question_id"]),
                "image_path": cand,
                "modality_type": it.get("modality_type", ""),
                "question_type": it.get("question_type", ""),
                "question": it.get("question", ""),
                "options": opts,
                "gt_letter": gt_letter,
                "gt_answer": gt_raw,
            })

    if args.max_items and args.max_items > 0:
        items = items[:args.max_items]
        print(f"[Eval] Truncated to first {len(items)} items")

    device = "cuda:0"
    model, tokenizer = load_merged_model(args.base_model, args.adapter_path, device)
    P1, P2 = (NOISE_P1, NOISE_P2) if args.prompt_mode == "noise" else (PASSIVE_P1, PASSIVE_P2)

    preds_path = os.path.join(args.out_dir, "sub2000_preds.jsonl")
    done_map = {}
    if os.path.exists(preds_path):
        for line in open(preds_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done_map[r["question_id"]] = r
        print(f"[Eval] Resuming: {len(done_map)}/{len(items)} already done")

    with open(preds_path, "a", encoding="utf-8") as fout:
        for idx, it in enumerate(items):
            if it["question_id"] in done_map:
                continue
            im = Image.open(it["image_path"]).convert("RGB")
            opt_text = "\n".join(f"{l}. {t}" for l, t in it["options"])
            q = P1 + f"\n\nQuestion:\n{it['question']}\nOptions:\n{opt_text}\nAnswer:\n" + P2
            inp = model.build_conversation_input_ids(tokenizer=tokenizer, query=q, history=None, images=[im])
            gen_in = {
                "input_ids": inp["input_ids"].unsqueeze(0).to(device),
                "token_type_ids": inp["token_type_ids"].unsqueeze(0).to(device),
                "attention_mask": inp["attention_mask"].unsqueeze(0).to(device),
                "images": [[inp["images"][0].to(device, dtype=torch.bfloat16)]],
            }
            with torch.no_grad():
                out = model.generate(**gen_in, max_new_tokens=128, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or 128002)
            pred_text = tokenizer.decode(out[0, gen_in["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            pl = extract_letter(pred_text)
            ok = 1.0 if (pl == it["gt_letter"]) else 0.0
            ct = strip_noise(pred_text)
            rec = {
                "question_id": it["question_id"],
                "modality_type": it["modality_type"],
                "question_type": it["question_type"],
                "gt_letter": it["gt_letter"],
                "pred_letter": pl,
                "correct": bool(ok),
                "pred_text": pred_text,
                "pred_text_clean": ct,
            }
            done_map[it["question_id"]] = rec
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if len(done_map) % 200 == 0:
                cur_acc = sum(1 for r in done_map.values() if r["correct"]) / len(done_map)
                print(f"  [Eval] {len(done_map)}/{len(items)} acc={cur_acc*100:.2f}%", flush=True)

    acc_vals = [1.0 if done_map[it["question_id"]]["correct"] else 0.0 for it in items]
    clean_texts = [done_map[it["question_id"]]["pred_text_clean"] for it in items]
    unparseable = sum(1 for it in items if not done_map[it["question_id"]]["pred_letter"])
    l0_hits = sum(
        1 for it in items
        if (_NOISE_LVL_RE.search(done_map[it["question_id"]]["pred_text"]) and int(_NOISE_LVL_RE.search(done_map[it["question_id"]]["pred_text"]).group(1)) == 0)
    )

    del model, tokenizer
    torch.cuda.empty_cache()

    print("[Eval] Scoring with BMCA, PLIP, UniMed-CLIP...")
    vds_res = score_3clip(items, clean_texts, device)
    acc = sum(acc_vals) / len(acc_vals)
    ci = cluster_ci95(items, acc_vals)

    summary = {
        "n": len(items),
        "acc": acc,
        "acc_ci95": ci,
        "unparseable": unparseable,
        "l0_pred_rate": l0_hits / len(items),
        "vds": vds_res,
    }
    out_json = os.path.join(args.out_dir, "sub2000_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Eval] SUMMARY: Acc={acc*100:.2f}% [{ci[0]*100:.2f},{ci[1]*100:.2f}] | BMCA={vds_res['bmca']['paired']:.4f} (gap_sm={vds_res['bmca']['gap_same_mod']:+.4f}) | PLIP={vds_res['plip']['paired']:.4f} | UniMed={vds_res['unimed']['paired']:.4f} -> {out_json}")


if __name__ == "__main__":
    main()
