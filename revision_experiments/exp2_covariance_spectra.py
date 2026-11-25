# -*- coding: utf-8 -*-
"""
Exp P0-2: Visual Embedding Covariance Spectra Before vs. After NAP Training
Directly tests Theoretical Section 4 (Eq. 9):
  I_eff(t) = (d / 2) * Var_i[ 1 / (lambda_i + t) ]
on visual token matrices V (n x d) produced by the EVA-CLIP-E + adapter right before
entering the language stream, on a stratified 200-image subset of Sub2000.
"""
import os
import json
import random
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda:0"
OUT_DIR = "/path/to/revision_outputs/exp2_spectra"
os.makedirs(OUT_DIR, exist_ok=True)

SUB2000_ROOT = "/path/to/datasets/OmniMedVQA-Clean-Sub2000"
qa_dir = os.path.join(SUB2000_ROOT, "QA_information", "Open-access")

by_mod = defaultdict(list)
for fn in sorted(os.listdir(qa_dir)):
    if not fn.endswith(".json"):
        continue
    for it in json.load(open(os.path.join(qa_dir, fn), encoding="utf-8")):
        ip = it["image_path"]
        cand = ip if os.path.isabs(ip) else os.path.join(SUB2000_ROOT, ip)
        if not os.path.exists(cand):
            cand = os.path.join(SUB2000_ROOT, "Images", ip)
        by_mod[it.get("modality_type", "")].append((str(it["question_id"]), cand, it.get("modality_type", "")))

# Pick 25 unique images per modality = 200 images (seed 42)
rng = random.Random(42)
selected = []
for mod in sorted(by_mod.keys()):
    seen = set()
    uniq = []
    for qid, p, m in by_mod[mod]:
        if p not in seen:
            seen.add(p)
            uniq.append((qid, p, m))
    rng.shuffle(uniq)
    selected.extend(uniq[:25])

print(f"[Exp2] Selected {len(selected)} images across {len(by_mod)} modalities")

NOISE_ALPHAS = {1: 0.072, 2: 0.144, 3: 0.216, 4: 0.288, 5: 0.360}

MODELS = {
    "CogVLM2-SFT": "/path/to/models/THUDM/vlcs/cogvlm2-llama3-chat-vlcs-19B-sft-plus",
    "w/o_ST+R_np": "/path/to/models/THUDM/vlcs/plus/cogvlm2-llama3-chat-vlcs-19B-vdrfv2_non-2000",
    "w/o_R_np": "/path/to/models/THUDM/vlcs2/cogvlm2-llama3-chat-vlcs-19B-sft-medreason2026-vdrfv2-nnp-300",
    "NAP-19B": "/path/to/models/THUDM/vlcs2/cogvlm2-llama3-chat-vlcs-19B-sft-medreason2026-vdrfv2-400",
}

from torchvision import transforms

def preprocess_cogvlm_image(pil_img, image_size=1344):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    return tfm(pil_img)

pil_imgs = [Image.open(p).convert("RGB") for _, p, _ in selected]

all_results = {}

for mname, mpath in MODELS.items():
    print(f"\n[Exp2] Loading {mname} from {mpath} ...")
    model = AutoModelForCausalLM.from_pretrained(
        mpath, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()
    # Move ONLY the vision module (EVA-CLIP-E + adapter linear_proj) to GPU 0!
    vision = model.model.vision.to(DEVICE).eval()
    image_size = model.config.vision_config.get("image_size", 1344)
    print(f"  vision loaded on {DEVICE}, image_size={image_size}")

    ieff_by_level = {l: [] for l in range(1, 6)}
    eff_ranks = []
    top1_ratios = []
    top10_ratios = []
    token_means = []
    token_stds = []
    kurtoses = []
    eig_profiles = []  # store top-64 eigenvalues averaged across images

    with torch.no_grad():
        for idx, im in enumerate(pil_imgs):
            x = preprocess_cogvlm_image(im, image_size).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
            V = vision(x)  # (1, n, d)
            V = V.squeeze(0).float()  # (n, d) on GPU
            n, d = V.shape

            # Basic distributional checks (zero-mean & Gaussianity check for R3.8)
            mean_abs = float(V.mean(dim=0).abs().mean().item())
            std_all = float(V.std().item())
            z = (V - V.mean()) / (V.std() + 1e-8)
            kurt = float((z ** 4).mean().item())  # Gaussian = 3.0
            token_means.append(mean_abs / (std_all + 1e-8))
            token_stds.append(std_all)
            kurtoses.append(kurt)

            # Second moment Sigma = (1/n) V^T V (d x d), normalized so tr(Sigma_1)/d = 1
            # Compute eigenvalues via Gram matrix G = (1/n) V V^T (n x n) since n=2304 < d=4096
            G = (V @ V.T) / float(n)
            eigs_n = torch.linalg.eigvalsh(G).clamp(min=0.0)  # ascending, length n
            # Full d eigenvalues = eigs_n padded with (d - n) zeros (if n < d)
            if d > n:
                eigs_d = torch.cat([torch.zeros(d - n, device=V.device), eigs_n])
            else:
                eigs_d = eigs_n[-d:]
            c_scale = eigs_d.mean().item()  # tr(Sigma)/d
            lam = (eigs_d / (c_scale + 1e-12)).cpu().numpy()  # normalized eigenvalues, mean=1

            # Theoretical Eq. 9: I_eff(t) = (d/2) * Var_i[ 1 / (lam_i + t) ]
            for l, alpha in NOISE_ALPHAS.items():
                t = alpha ** 2
                mu = 1.0 / (lam + t)
                ieff = 0.5 * d * float(np.var(mu))
                ieff_by_level[l].append(ieff)

            # Spectral concentration metrics
            lam_desc = lam[::-1]
            p_eig = lam_desc / (lam_desc.sum() + 1e-12)
            p_pos = p_eig[p_eig > 1e-12]
            ent = -float(np.sum(p_pos * np.log(p_pos)))
            eff_ranks.append(float(np.exp(ent)))
            top1_ratios.append(float(p_eig[0]))
            top10_ratios.append(float(p_eig[:10].sum()))
            eig_profiles.append(lam_desc[:64])

    avg_profile = np.mean(eig_profiles, axis=0).tolist()
    res = {
        "n_images": len(pil_imgs),
        "n_tokens": int(n),
        "dim": int(d),
        "norm_mean_abs_over_std": float(np.mean(token_means)),
        "mean_std": float(np.mean(token_stds)),
        "mean_kurtosis": float(np.mean(kurtoses)),
        "effective_rank_mean": float(np.mean(eff_ranks)),
        "effective_rank_std": float(np.std(eff_ranks)),
        "top1_energy_ratio": float(np.mean(top1_ratios)),
        "top10_energy_ratio": float(np.mean(top10_ratios)),
        "I_eff_by_level": {
            f"L{l}": {
                "alpha": NOISE_ALPHAS[l],
                "t": NOISE_ALPHAS[l] ** 2,
                "mean": float(np.mean(ieff_by_level[l])),
                "std": float(np.std(ieff_by_level[l])),
            }
            for l in range(1, 6)
        },
        "avg_top64_eigenvalues": avg_profile,
    }
    all_results[mname] = res
    print(
        f"  {mname:12s} | eff_rank={res['effective_rank_mean']:.2f}±{res['effective_rank_std']:.2f} "
        f"| top1={res['top1_energy_ratio']:.4f} top10={res['top10_energy_ratio']:.4f} "
        f"| I_eff(L1)={res['I_eff_by_level']['L1']['mean']:.1f} I_eff(L3)={res['I_eff_by_level']['L3']['mean']:.1f} I_eff(L5)={res['I_eff_by_level']['L5']['mean']:.1f} "
        f"| kurt={res['mean_kurtosis']:.2f} |mean|/std={res['norm_mean_abs_over_std']:.4f}"
    )

    del vision, model
    torch.cuda.empty_cache()

out_path = os.path.join(OUT_DIR, "covariance_spectra_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\n[Exp2] DONE -> {out_path}")
