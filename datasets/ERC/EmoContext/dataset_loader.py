from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class TrainValidDataset(Dataset):
    def __init__(self, df, tokenizer_name, max_len):
        self.df = df
        self.text = df["text"].values
        self.target = df["target"].values
        self.tokenizer = AutoTokenizer(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        texts = self.text[idx]
        tokenized = self.tokenizer.encode_plus(texts, truncation=True, add_special_tokens=True,
                                               max_length=self.max_len, padding="max_length")
        ids = tokenized["input_ids"]
        mask = tokenized["attention_mask"]
        targets = self.target[idx]
        return {
            "ids": torch.LongTensor(ids),
            "mask": torch.LongTensor(mask),
            "targets": torch.tensor(targets, dtype=torch.float32)
        }