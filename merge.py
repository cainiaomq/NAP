from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from safetensors.torch import load_file, save_file
import torch, os, json, shutil, tempfile

base_model_path = "checkpoints/sft"
lora_model_path = "checkpoints/nap"
output_path = "checkpoints/nap"

model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

sd_path = os.path.join(lora_model_path, "adapter_model.safetensors")
if os.path.exists(sd_path):
    sd = load_file(sd_path)
else:
    sd = torch.load(os.path.join(lora_model_path, "adapter_model.bin"), map_location="cpu")

lora_sd = {}
proj_sd = {}
for k, v in sd.items():
    if "linear_proj" in k:
        new_k = k.replace("base_model.model.", "").replace("linear_proj.linear_proj", "linear_proj")
        proj_sd[new_k] = v
        print(f"[proj] {k} → {new_k}")
    else:
        lora_sd[k] = v

tmp_dir = tempfile.mkdtemp()
shutil.copy(os.path.join(lora_model_path, "adapter_config.json"), tmp_dir)

with open(os.path.join(tmp_dir, "adapter_config.json"), "r") as f:
    cfg = json.load(f)
cfg["modules_to_save"] = None
with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
    json.dump(cfg, f)

save_file(lora_sd, os.path.join(tmp_dir, "adapter_model.safetensors"))

model = PeftModel.from_pretrained(model, tmp_dir)
model = model.merge_and_unload()
print("[OK] LoRA merged")

missing, unexpected = model.load_state_dict(proj_sd, strict=False)
print(f"[OK] linear_proj loaded | missing={len(missing)} unexpected={len(unexpected)}")

model.save_pretrained(output_path, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_path)

shutil.rmtree(tmp_dir)
print(f"[Done] saved to {output_path}")
