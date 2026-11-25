# -*- coding: utf-8 -*-
"""
R3.9 Clean Control 1 (Stage 1): Online Clean Self-Training WITHOUT noise injection or <noise> tags.
Strictly samples 400 questions from OmniMedVQA-Train (zero overlap with OmniMedVQA-Clean/Sub2000),
generates K=4 candidates on clean images (L0), selects the correct candidate with highest BiomedCLIP
similarity (falling back to GT on L0 if all 4 are wrong), trains with noise_level=0 and NO <noise> tags,
and merges LoRA + linear_proj into --merged_out for Stage 2 (1,000-step Clean GRPO).
"""
import argparse, gc, json, os, re, random, shutil, sys, tempfile, time, logging
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import HfDeepSpeedConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from safetensors.torch import load_file, save_file

sys.path.insert(0, "/path/to/NAP_code")
from utils.omnimedqkv import OmniMedVQA_Dataset
from rl.embedder import MedClipEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("r39-clean-st")

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


def build_train_sample(model, tokenizer, image_pil, question, options, label_text, fallback, max_length, device, dtype):
    opt_text = "\n".join(f"{l}. {t}" for l, t in options)
    if fallback:
        query = f"Question:\n{question}\nOptions:\n{opt_text}\nAnswer:\n"
    else:
        query = PASSIVE_P1 + f"\n\nQuestion:\n{question}\nOptions:\n{opt_text}\nAnswer:\n" + PASSIVE_P2
    inp = model.build_conversation_input_ids(
        tokenizer=tokenizer, query=query, history=None, images=[image_pil], answer=label_text
    )

    def pad_to_len(t, length, pad_value=0):
        l = len(t)
        return t[:length] if l >= length else torch.cat(
            (t, torch.full([length - l], fill_value=pad_value, dtype=t.dtype, device=t.device)), dim=0
        )

    inp["input_ids"] = pad_to_len(inp["input_ids"], max_length, pad_value=128002).unsqueeze(0).to(device)
    inp["attention_mask"] = pad_to_len(inp["attention_mask"], max_length, pad_value=0).unsqueeze(0).to(device)
    inp["token_type_ids"] = pad_to_len(inp["token_type_ids"], max_length, pad_value=0).unsqueeze(0).to(device)
    inp["labels"] = pad_to_len(inp["labels"], max_length, pad_value=-100).unsqueeze(0).to(device)
    inp["images"] = [[img.to(device).to(dtype) for img in inp["images"]]]
    inp["noise_level"] = torch.tensor([0], dtype=torch.long, device=device)
    return inp


def merge_and_save(base_path, adapter_path, merged_out):
    logger.info(f"[Merge] Merging {adapter_path} into {base_path} -> {merged_out}")
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
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
    logger.info(f"[Merge] proj_keys={len(proj_sd)} missing={len(missing)} unexpected={len(unexpected)}")
    os.makedirs(merged_out, exist_ok=True)
    model.save_pretrained(merged_out, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_out)
    for fn in os.listdir(base_path):
        if fn.endswith(".py"):
            shutil.copy(os.path.join(base_path, fn), os.path.join(merged_out, fn))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"[Merge] Saved merged Clean-ST model -> {merged_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset_path", default="/path/to/datasets/OmniMedVQA-Train")
    ap.add_argument("--ds_config", default="/path/to/NAP_code/ds_config.yaml")
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--merged_out", required=True)
    ap.add_argument("--max_samples", type=int, default=400)
    ap.add_argument("--num_candidates", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local_rank", type=int, default=-1)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.ds_config, "r") as f:
        ds_cfg = yaml.safe_load(f)
    accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=HfDeepSpeedConfig(ds_cfg)))
    device = accelerator.device
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True)

    # Load strictly from OmniMedVQA-Train with identical HVD filtering
    dataset = OmniMedVQA_Dataset(
        root_dir=args.dataset_path, tokenizer=tokenizer, model=base,
        torch_type=dtype, input_length=640, output_length=320,
        noise_rate=0.0, use_doctor_prompt=True,
        visdep_sidecar="/path/to/datasets/OmniMedVQA/iodep_scores.jsonl",
        visdep_min_score=0.35,
        dep_sidecar="/path/to/datasets/OmniMedVQA/visdep_scores.jsonl",
        dep_min=0.0,
    )
    samples = list(dataset.samples)
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    selected = samples[:args.max_samples]
    logger.info(f"[Clean-ST] Sampled {len(selected)} strictly OmniMedVQA-Train items (pool={len(samples)})")

    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=[
            "language_expert_query_key_value", "language_expert_dense",
            "language_mlp.gate_proj", "language_mlp.up_proj", "language_mlp.down_proj",
        ],
        inference_mode=False, modules_to_save=["linear_proj"],
    )
    model = get_peft_model(base, lcfg)
    embedder = MedClipEmbedder(); embedder.eval()

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=len(selected) + 10)
    model, optim, lr_scheduler = accelerator.prepare(model, optim, lr_scheduler)

    os.makedirs(args.save_path, exist_ok=True)
    audit_path = os.path.join(args.save_path, "clean_st_audit.jsonl")
    max_length = 640 + 320 + 2303
    unwrapped = accelerator.unwrap_model(model)

    n_distilled = 0
    n_fallback = 0
    losses = []

    with open(audit_path, "w", encoding="utf-8") as af:
        for idx, s in enumerate(tqdm(selected, desc="Online Clean-ST")):
            im = Image.open(s["image_path"]).convert("RGB")
            options = s.get("options", [])
            raw_gt = s.get("gt_answer", "")
            gt_with_letter = raw_gt
            for ltr, opt in options:
                if opt.strip().lower().rstrip(".") == raw_gt.strip().lower().rstrip("."):
                    gt_with_letter = f"{ltr}. {opt}"
                    break
            gt_letter = extract_letter(gt_with_letter)
            opt_text = "\n".join(f"{l}. {t}" for l, t in options)
            query = PASSIVE_P1 + f"\n\nQuestion:\n{s['question']}\nOptions:\n{opt_text}\nAnswer:\n" + PASSIVE_P2

            # Step A: generate K=4 clean candidates
            model.eval()
            inp = unwrapped.build_conversation_input_ids(tokenizer=tokenizer, query=query, history=None, images=[im])
            gen_in = {
                "input_ids": inp["input_ids"].unsqueeze(0).to(device),
                "token_type_ids": inp["token_type_ids"].unsqueeze(0).to(device),
                "attention_mask": inp["attention_mask"].unsqueeze(0).to(device),
                "images": [[inp["images"][0].to(device, dtype=dtype)]],
            }
            cands = []
            with torch.no_grad():
                for _ in range(args.num_candidates):
                    out = unwrapped.generate(
                        **gen_in, max_new_tokens=160, do_sample=True, top_p=0.95, temperature=1.0,
                        pad_token_id=tokenizer.pad_token_id or 128002,
                    )
                    txt = tokenizer.decode(out[0, gen_in["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    txt_clean = _NOISE_TAG_RE.sub("", txt).strip()
                    cands.append(txt_clean)

                # Step B: BiomedCLIP VDS selection among correct candidates
                ie = F.normalize(embedder.encode_image([im]).float(), dim=-1)
                te = F.normalize(embedder.encode_text([c or "empty" for c in cands]).float(), dim=-1)
                vds_scores = (ie * te).sum(-1).cpu().tolist()

            correct_cands = [(vds_scores[j], cands[j]) for j in range(len(cands)) if extract_letter(cands[j]) == gt_letter]
            if correct_cands:
                correct_cands.sort(key=lambda x: -x[0])
                chosen_text = correct_cands[0][1]
                is_fb = False
                n_distilled += 1
            else:
                chosen_text = gt_with_letter.strip()
                is_fb = True
                n_fallback += 1

            # Step C: Clean SFT update (noise_level=0, NO <noise> tags)
            model.train()
            mb = build_train_sample(unwrapped, tokenizer, im, s["question"], options, chosen_text, is_fb, max_length, device, dtype)
            out = model(
                input_ids=mb["input_ids"], token_type_ids=mb["token_type_ids"],
                attention_mask=mb["attention_mask"], images=mb["images"],
                labels=mb["labels"], noise_level=mb["noise_level"],
            )
            loss = out.loss
            optim.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optim.step(); lr_scheduler.step()
            lv = float(loss.detach().item())
            losses.append(lv)
            del out, loss, mb

            af.write(json.dumps({
                "idx": idx + 1, "qid": s.get("question_id"), "modality_type": s.get("modality_type"),
                "question_type": s.get("question_type"), "fallback": is_fb,
                "chosen_text": chosen_text, "loss": lv,
            }, ensure_ascii=False) + "\n")
            af.flush()

            if (idx + 1) % 50 == 0:
                logger.info(f"[Clean-ST] {idx+1}/{len(selected)} distilled={n_distilled} fb={n_fallback} mean_loss={sum(losses[-50:])/50:.4f}")

    adapter_dir = os.path.join(args.save_path, "clean_st_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(adapter_dir, safe_serialization=True)
    logger.info(f"[Clean-ST] DONE: distilled={n_distilled}, fallback={n_fallback} -> {adapter_dir}")

    del model, base, optim, lr_scheduler, embedder
    gc.collect()
    torch.cuda.empty_cache()

    merge_and_save(args.model_path, adapter_dir, args.merged_out)


if __name__ == "__main__":
    main()
