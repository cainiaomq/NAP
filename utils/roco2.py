import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset, DatasetDict


class ROCOv2_Dataset(Dataset):
    def __init__(self,
                 root_dir,
                 tokenizer,
                 model,
                 torch_type,
                 device='cuda',
                 input_length=1024,
                 output_length=1024,
                 noise_rate=0.0,
                 split='train'):
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

        data_path = os.path.join(self.root_dir, "data")
        self.dataset = load_dataset("parquet", data_dir=data_path, split=split)
        print(f"[ROCOv2_Dataset] loaded samples: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def custom_collate_fn(batch):
        batched_data = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], list):
                batched_data[key] = [item[key] for item in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                batched_data[key] = torch.stack([item[key] for item in batch])
            else:
                raise ValueError(f"Unsupported datatype for key {key} in custom collate_fn")
        return batched_data

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["image"]
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(image["path"] if "path" in image else image["bytes"]).convert("RGB")
        elif isinstance(image, str) and os.path.exists(image):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError(f"unsupported image type: {type(image)}")

        caption = sample["caption"]

        if random.random() < self.noise_rate:
            noise_level = random.randint(1, 5)
        else:
            noise_level = 0
        answer_with_noise = f"<noise>L{noise_level}</noise> " + caption

        input_data = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query="Please describe the medical image.",
            history=None,
            images=[image],
            answer=answer_with_noise
        )

        def pad_to_len(unpadded_tensor, pad_to_length, pad_value=0):
            current_length = len(unpadded_tensor)
            if current_length >= pad_to_length:
                return unpadded_tensor[:pad_to_length]
            return torch.cat((
                unpadded_tensor,
                torch.full([pad_to_length - current_length],
                           fill_value=pad_value,
                           dtype=unpadded_tensor.dtype,
                           device=unpadded_tensor.device)), dim=0)

        input_data['input_ids'] = pad_to_len(input_data['input_ids'], self.max_length, pad_value=128002)
        input_data['attention_mask'] = pad_to_len(input_data['attention_mask'], self.max_length, pad_value=0)
        input_data['token_type_ids'] = pad_to_len(input_data['token_type_ids'], self.max_length, pad_value=0)
        input_data['labels'] = pad_to_len(input_data['labels'], self.max_length, pad_value=-100)
        input_data['noise_level'] = torch.tensor(noise_level, dtype=torch.long)

        for key in input_data:
            if key == 'images':
                input_data[key] = [img.to(self.device).to(self.torch_type) for img in input_data[key]]
            else:
                input_data[key] = input_data[key].to(self.device)

        return input_data
