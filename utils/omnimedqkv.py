import os
import json
import random
import hashlib
import torch
from torch.utils.data import Dataset
from PIL import Image


class OmniMedVQA_Dataset(Dataset):
    

    def __init__(self,
                 root_dir,
                 tokenizer,
                 model,
                 torch_type,
                 device='cuda',
                 input_length=1024,
                 output_length=1024,
                 noise_rate=0.0,
                 use_doctor_prompt: bool = False,
                 access='open',
                 split_files=None,
                 visdep_sidecar: str = None,
                 visdep_min_score: float = None,
                 visdep_weighting: bool = False,
                 dep_sidecar: str = None,
                 dep_min: float = 0.0,
                 keep_if_dep_missing: bool = True,
                 fixed_noise_seed: int = None,
                 eval_noise_mode: str = "by_rate",
                 eval_min_level: int = 1,
                 eval_max_level: int = 5,
                 ):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.torch_type = torch_type
        self.input_length = input_length
        self.output_length = output_length
        self.noise_rate = noise_rate
        self.padding_len = 2303
        self.max_length = self.input_length + self.output_length + self.padding_len
        self.access = access
        self.split_files = split_files

        self.fixed_noise_seed = fixed_noise_seed
        self.eval_noise_mode = eval_noise_mode
        self.eval_min_level = int(eval_min_level)
        self.eval_max_level = int(eval_max_level)
        self._forced_level = None
        if isinstance(eval_noise_mode, str) and eval_noise_mode.startswith("all_L"):
            try:
                self._forced_level = int(eval_noise_mode[len("all_L"):])
                assert 0 <= self._forced_level <= 5
            except Exception as e:
                raise ValueError(f"failed to parse eval_noise_mode='{eval_noise_mode}': {e}")

        self._dep_map = self._load_dep_sidecar(dep_sidecar) if dep_sidecar else {}
        self.dep_min = dep_min
        self.keep_if_dep_missing = keep_if_dep_missing

        self.use_doctor_prompt = use_doctor_prompt
        self.doctor_prompt1 = (
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


        self.doctor_prompt2 = (
            "You MUST respond in EXACTLY this format on a single line, with no extra text:\n"
            "<noise>L<digit></noise> <letter>. <option text>: Evidence: <one concise sentence>\n\n"
            "Where:\n"
            "  - <digit> is one integer from 0 to 5 (your noise estimate). e.g., <noise>L3</noise>.\n"
            "  - <letter> is one of A, B, C, D (your chosen option)\n"
            "  - <option text> is the exact text of the chosen option\n"
            "  - Evidence is ONE concise sentence describing the key visual cue supporting your choice\n\n"
            "Output the answer line now:"
        )


        self._visdep_map = self._load_visdep_sidecar(visdep_sidecar) if visdep_sidecar else {}

        qa_base = os.path.join(root_dir, 'QA_information')
        self.qa_dirs = []
        if access in ['open', 'both']:
            self.qa_dirs.append(os.path.join(qa_base, 'Open-access'))
        if access == 'both':
            self.qa_dirs.append(os.path.join(qa_base, 'Restricted-access'))
        if not self.qa_dirs:
            raise ValueError(f"invalid access='{access}': no QA information path")

        self.visdep_min_score = visdep_min_score
        self.visdep_weighting = visdep_weighting

        self.samples = self._load_samples()
        if self.visdep_min_score is not None:
            before = len(self.samples)
            self.samples = [s for s in self.samples if s.get('visdep_score', 0.5) >= float(self.visdep_min_score)]
            print(f"[OmniMedVQA_Dataset] visual dependency filter: {before} -> {len(self.samples)} (threshold >= {self.visdep_min_score})")

        print(f"[OmniMedVQA_Dataset] loaded {len(self.samples)} samples")

        if self.dep_min is not None:
            before = len(self.samples)
            kept, dropped = [], 0
            for s in self.samples:
                dep = s.get('dep', None)
                if dep is None:
                    if self.keep_if_dep_missing:
                        kept.append(s)
                    else:
                        dropped += 1
                else:
                    if float(dep) >= float(self.dep_min):
                        kept.append(s)
                    else:
                        dropped += 1
            self.samples = kept
            print(f"[OmniMedVQA_Dataset] dep filter: {before} -> {len(self.samples)} (threshold >= {self.dep_min}, dropped {dropped})")

        if self.fixed_noise_seed is not None:
            print(f"[OmniMedVQA_Dataset] deterministic noise: seed={self.fixed_noise_seed}, "
                  f"mode={self.eval_noise_mode}, level_range=[{self.eval_min_level},{self.eval_max_level}], "
                  f"forced_level={self._forced_level}")

    def _load_visdep_sidecar(self, path):
        if not os.path.exists(path):
            print(f"[OmniMedVQA_Dataset] visdep_sidecar not found: {path}")
            return {}
        m = {}
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        obj = json.loads(line)
                        ip = obj.get("question_id")
                        sc = obj.get("visdep_score")
                        if ip is not None and isinstance(sc, (int, float)):
                            m[ip] = float(max(0.0, min(1.0, sc)))
                    except Exception:
                        continue
        else:
            try:
                data = json.load(open(path, "r", encoding="utf-8"))
                if isinstance(data, list):
                    for obj in data:
                        ip = obj.get("question_id")
                        sc = obj.get("visdep_score")
                        if ip is not None and isinstance(sc, (int, float)):
                            m[os.path.abspath(ip)] = float(max(0.0, min(1.0, sc)))
                elif isinstance(data, dict):
                    for ip, sc in data.items():
                        if isinstance(sc, (int, float)):
                            m[os.path.abspath(ip)] = float(max(0.0, min(1.0, sc)))
            except Exception as e:
                print(f"[OmniMedVQA_Dataset] failed to parse sidecar: {path} | {e}")
        print(f"[OmniMedVQA_Dataset] loaded visdep_sidecar entries: {len(m)}")
        return m

    def _load_dep_sidecar(self, path: str):
        if not os.path.exists(path):
            print(f"[OmniMedVQA_Dataset] dep_sidecar not found: {path}")
            return {}
        m = {}
        def _clip(x): return float(max(-1.0, min(1.0, float(x))))
        try:
            if path.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        qid = obj.get("question_id")
                        dep = obj.get("dep", obj.get("dep_score", None))
                        if dep is None:
                            vg = obj.get("vec_g", None)
                            vcf = obj.get("vec-cf", obj.get("vec_cf", None))
                            if vg is not None and vcf is not None:
                                dep = float(vg) - float(vcf)
                        if qid is not None and isinstance(dep, (int, float)):
                            m[str(qid)] = _clip(dep)
            else:
                data = json.load(open(path, "r", encoding="utf-8"))
                if isinstance(data, list):
                    for obj in data:
                        qid = obj.get("question_id")
                        dep = obj.get("dep", obj.get("dep_score", None))
                        if dep is None:
                            vg = obj.get("vec_g", None)
                            vcf = obj.get("vec-cf", obj.get("vec_cf", None))
                            if vg is not None and vcf is not None:
                                dep = float(vg) - float(vcf)
                        if qid is not None and isinstance(dep, (int, float)):
                            m[str(qid)] = _clip(dep)
                elif isinstance(data, dict):
                    for qid, dep in data.items():
                        if isinstance(dep, (int, float)):
                            m[str(qid)] = _clip(dep)
            print(f"[OmniMedVQA_Dataset] loaded dep_sidecar entries: {len(m)}")
        except Exception as e:
            print(f"[OmniMedVQA_Dataset] failed to parse dep_sidecar: {path} | {e}")
        return m

    def _load_samples(self):
        samples = []
        for qa_dir in self.qa_dirs:
            if not os.path.isdir(qa_dir):
                print(f"[OmniMedVQA_Dataset] QA directory not found: {qa_dir}")
                continue
            for fname in os.listdir(qa_dir):
                if not fname.endswith('.json'):
                    continue
                dataset_name = fname[:-5]
                if self.split_files and dataset_name not in self.split_files:
                    continue
                path = os.path.join(qa_dir, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"[OmniMedVQA_Dataset] failed to parse JSON: {path} | {e}")
                    continue

                for item in data:
                    img_rel = item.get('image_path')
                    if img_rel is None:
                        continue
                    img_abs = os.path.join(self.root_dir, img_rel)
                    if not os.path.exists(img_abs):
                        img_abs = os.path.join(self.root_dir, 'Images', img_rel)
                    if not os.path.exists(img_abs):
                        print(f"[Missing Image] {img_abs} not found, skipping")
                        continue
                    img_abs = os.path.abspath(img_abs)

                    vscore = item.get('visdep_score', None)
                    if not isinstance(vscore, (int, float)):
                        vscore = self._visdep_map.get(item.get('question_id'), 0.5)
                    vscore = float(max(0.0, min(1.0, vscore)))

                    samples.append({
                        'image_path': img_abs,
                        'question': item.get('question'),
                        'options': [(k[-1], item.get(k)) for k in ['option_A', 'option_B', 'option_C', 'option_D'] if item.get(k)],
                        'gt_answer': item.get('gt_answer'),
                        'question_type': item.get('question_type'),
                        'modality_type': item.get('modality_type'),
                        'dataset_name': item.get('dataset'),
                        'visdep_score': vscore,
                        'question_id': item.get('question_id'),
                        'dep': self._dep_map.get(str(item.get('question_id')), None),
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def custom_collate_fn(batch):
        batched = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], list):
                batched[key] = [x[key] for x in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                batched[key] = torch.stack([x[key] for x in batch])
            else:
                batched[key] = [x[key] for x in batch]
        return batched

    def _deterministic_noise_level(self, question_id) -> int:
        
        if self._forced_level is not None:
            return self._forced_level

        qid_str = str(question_id) if question_id is not None else ""
        h = hashlib.sha1(f"{qid_str}|{self.fixed_noise_seed}".encode("utf-8")).hexdigest()
        local_seed = int(h[:16], 16)
        rng = random.Random(local_seed)

        if self.eval_noise_mode == "always_random":
            return rng.randint(self.eval_min_level, self.eval_max_level)
        if rng.random() < self.noise_rate:
            return rng.randint(self.eval_min_level, self.eval_max_level)
        return 0

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s['image_path']).convert("RGB")
        question = s['question']
        options = s.get('options', [])
        raw_gt = s.get('gt_answer', '')

        gt = raw_gt
        for letter, opt in options:
            if opt.strip().lower().rstrip(".") == raw_gt.strip().lower().rstrip("."):
                gt = f"{letter}. {opt}"
                break

        opt_text = "\n".join(f"{letter}. {opt}" for letter, opt in options)
        prefix1 = (self.doctor_prompt1 + "\n") if self.use_doctor_prompt else ""
        prefix2 = (self.doctor_prompt2 + "\n") if self.use_doctor_prompt else ""
        query = prefix1 + f"\nQuestion:\n{question}\nOptions:\n{opt_text}\nAnswer:\n" + prefix2

        if self.fixed_noise_seed is not None:
            noise_level = self._deterministic_noise_level(s.get('question_id'))
        else:
            if random.random() < self.noise_rate:
                noise_level = random.randint(1, 5)
            else:
                noise_level = 0

        gt = f"<noise>L{noise_level}</noise> " + gt

        input_data = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=query,
            history=None,
            images=[image],
            answer=gt
        )

        def pad_to_len(t, length, pad_value=0):
            l = len(t)
            return t[:length] if l >= length else torch.cat(
                (t, torch.full([length - l], fill_value=pad_value, dtype=t.dtype, device=t.device)), dim=0
            )

        input_data['input_ids'] = pad_to_len(input_data['input_ids'], self.max_length, pad_value=128002)
        input_data['attention_mask'] = pad_to_len(input_data['attention_mask'], self.max_length, pad_value=0)
        input_data['token_type_ids'] = pad_to_len(input_data['token_type_ids'], self.max_length, pad_value=0)
        input_data['labels'] = pad_to_len(input_data['labels'], self.max_length, pad_value=-100)
        input_data['noise_level'] = torch.tensor(noise_level, dtype=torch.long)

        norm = lambda x: " ".join((x or "").strip().lower().rstrip(".。").split())
        g = norm(gt)
        input_data['answer_text'] = g
        input_data['options_text'] = opt_text
        vs = float(s['visdep_score'])
        input_data['visdep_score'] = torch.tensor(vs, device=self.device, dtype=torch.float32)
        if self.visdep_weighting:
            input_data['visdep_weight'] = torch.tensor(0.5 + 0.5 * vs, device=self.device, dtype=torch.float32)

        for key in input_data:
            if key == 'images':
                input_data[key] = [img.to(self.device).to(self.torch_type) for img in input_data[key]]
            else:
                if isinstance(input_data[key], torch.Tensor):
                    input_data[key] = input_data[key].to(self.device)

        question_id = s['question_id']
        question_type = s['question_type']
        modality_type = s['modality_type']
        input_data['question_id'] = question_id
        input_data['question_type'] = question_type
        input_data['modality_type'] = modality_type
        input_data['image_path'] = s['image_path']
        input_data['question'] = question
        input_data['raw_gt_answer'] = raw_gt
        input_data['gt_answer_with_letter'] = gt

        return input_data
