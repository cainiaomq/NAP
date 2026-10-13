# NAP: Noise-Aware Perception for Medical Vision-Language Models

Reference implementation of **Noise-Aware Perception: Supervising the Visual Pathway of
Medical Vision-Language Models through Noise-Level Prediction** (paper under review).

This work extends [Med-CogVLM](https://github.com/cainiaomq/Med-CogVLM) (PRICAI 2026).
Med-CogVLM constrains a deep-fusion medical VLM through visual dependency rewards on its
generated text; NAP gives the visual pathway itself a training objective, the prediction
of the noise level injected into the visual embedding.

NAP injects graded Gaussian noise into the visual embedding and trains the model
to report the noise level alongside its answer, using noise prediction as a
self-supervised perceptual objective for the visual pathway.

## Requirements

```bash
git clone https://github.com/cainiaomq/NAP.git
cd NAP
pip install -r requirements.txt
```

Experiments were run on a single A800 (80GB) with DeepSpeed ZeRO-2.

## Layout

```
model/modeling_cogvlm.py    CogVLM2 modeling file with embedding-level noise injection
lora_finetune.py            stage 1: progressive SFT
lora_finetune_st.py         stage 2: self-training of noise annotation
peft_grpo_duallora_v2.py    stage 3: noise-aware GRPO
merge.py                    merge LoRA weights into the base model
rl/                         reward functions and the medical CLIP embedder
utils/                      dataset readers (ROCO, ROCOv2, SLAKE, OmniMedVQA, CogCoM)
scripts/                    shallow-fusion counterparts used in the architecture comparison
ds_config.yaml              DeepSpeed configuration
```

## Noise injection

`model/modeling_cogvlm.py` replaces the corresponding file in the CogVLM2
checkpoint. Noise is added to the visual tokens produced by the encoder, before
they enter the language stream:

```
V = V + alpha * std(V) * eps,    eps ~ N(0, I)
```

`NOISE_LEVELS` maps each level to `alpha`. Scaling by the standard deviation of
the visual tokens keeps a level invariant to the scale of the representation.
The level is passed through the `noise_level` argument of `forward`.

## Training

Data directories are configured through `--dataset_path` and `--dataset_name`;
model and output locations through `--model_path` and `--save_path`.

**Stage 1 — progressive SFT.** Run once per curriculum stage, each initialised
from the previous one:

```bash
deepspeed lora_finetune.py --ds_config ds_config.yaml --dataset_name roco       --dataset_path data/ROCO       --save_path checkpoints/sft-1
deepspeed lora_finetune.py --ds_config ds_config.yaml --dataset_name rocov2     --dataset_path data/ROCOv2     --save_path checkpoints/sft-2 --resume_from checkpoints/sft-1
deepspeed lora_finetune.py --ds_config ds_config.yaml --dataset_name slake      --dataset_path data/SLAKE      --save_path checkpoints/sft-3 --resume_from checkpoints/sft-2
deepspeed lora_finetune.py --ds_config ds_config.yaml --dataset_name cogcom     --dataset_path data/CogCoM     --save_path checkpoints/sft-4 --resume_from checkpoints/sft-3
deepspeed lora_finetune.py --ds_config ds_config.yaml --dataset_name omnimedvqa --dataset_path data/OmniMedVQA-Train --save_path checkpoints/sft   --resume_from checkpoints/sft-4
python merge.py
```

**Stage 2 — self-training.** The model samples `--num_candidates` responses per
question, keeps the correct one with the highest visual dependency score, and
pairs it with a randomly drawn noise level. `--use_vds False` retains a random
correct candidate instead.

```bash
deepspeed lora_finetune_st.py --ds_config ds_config.yaml \
    --model_path checkpoints/sft \
    --dataset_path data/OmniMedVQA-Train \
    --save_path checkpoints/st \
    --num_candidates 4
```

**Stage 3 — noise-aware GRPO.** Each input is noised with probability
`--noise_rate`. The reward combines answer accuracy, output format, noise-level
prediction, and visual consistency against the clean image.

```bash
deepspeed peft_grpo_duallora_v2.py --ds_config ds_config.yaml \
    --model_path checkpoints/st \
    --dataset_path data/OmniMedVQA-Train \
    --save_path checkpoints/nap \
    --k 4 --noise_rate 0.5
```

Ablations reported in the paper:

| Configuration | Setting |
|---|---|
| w/o R_vec | `--w_vec 0` |
| w/o R_np | `--w_noise_pred 0` |
| w/o ST + R_np | `--w_noise_pred 0`, with `--model_path checkpoints/sft` (Stage 2 skipped) |

## Evaluation

`eval_omnimedvqa.py` reports accuracy, noise-level prediction accuracy, and VDS.
`--eval_noise_mode by_rate` evaluates on clean inputs at `--noise_rate 0`;
`always_random` and `all_Lx` evaluate under noise. `--fixed_noise_seed` makes the
per-question noise level deterministic.

```bash
python eval_omnimedvqa.py \
    --model_path checkpoints/nap \
    --dataset_path data/OmniMedVQA \
    --out_dir eval_out \
    --eval_noise_mode by_rate --noise_rate 0
```

## Architecture comparison

`scripts/` contains the same two stages for a shallow-fusion model, in which
noise is injected by a forward hook on the visual encoder rather than inside the
modeling file.

```bash
deepspeed scripts/lora_finetune_st_lingshu.py --ds_config ds_config.yaml
deepspeed scripts/peft_grpo_duallora_v2_lingshu.py --ds_config ds_config.yaml
```

## Related work

- **Med-CogVLM** (PRICAI 2026): [github.com/cainiaomq/Med-CogVLM](https://github.com/cainiaomq/Med-CogVLM)

## License

This project is licensed under the Apache License 2.0.
