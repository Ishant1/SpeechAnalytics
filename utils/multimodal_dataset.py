import re
import string
from typing import Optional

import pandas as pd
import torch
import torchaudio
import transformers
from speechbrain.dataio.preprocess import AudioNormalizer
from torch.utils.data import Dataset


class TextAudioDataset(Dataset):
    """Text and audio dataset for multimodal model, using BERT and wav2vec2."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        target_encoder,
        processor: transformers.Wav2Vec2Processor,
        max_len: Optional[int],
    ):
        """
        Args:
            :param df: dataframe of utterances and emotion label
            :param tokenizer: tokenizer for creating embedding of utterances
            :param target_encoder: encoding of text labels to integers e.g. instance of OneHotEncoder
            :param processor: processor loaded from a pretrained model e.g.
                              Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            :param max_len: maximum length of embedding
        """
        self.df = df
        self.text = df["text"].values
        self.audio_filepath = df["filepath"].values
        self.target = df["target"].values
        self.tokenizer = tokenizer
        self.target_encoder = target_encoder
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        """This method means that len(dataset) returns the size of the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Supports the indexing such that dataset[i] can be used to get the ith sample."""
        text = simple_text_clean(self.text[idx])
        tokenized = self.tokenizer.encode_plus(
            text=text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        ids = tokenized["input_ids"]
        mask = tokenized["attention_mask"]

        wav_filename = self.audio_filepath[idx]
        audio = torchaudio.load(wav_filename)
        raw_audio, sampling_rate = audio

        an = AudioNormalizer()
        raw_audio = an(raw_audio.unsqueeze(0), sampling_rate)

        input_values = self.processor(
            audio=raw_audio, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_values

        targets = (
            self.target_encoder.transform([[self.target[idx]]]).toarray().reshape(-1)
        )

        return {
            "text": {
                "input_ids": ids,
                "mask": mask,
            },
            "audio": {"input_values": input_values},
            "targets": torch.tensor(targets, dtype=torch.float32),
        }


def simple_text_clean(text: str) -> str:
    """Function to clean text data.

    Args:
        text: string of text to be cleaned
    Return:
        string of text with unwanted characters removed
    """
    text = text.lower()
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"https*\S+", " ", text)
    text = re.sub(r"http*\S+", " ", text)
    text = re.sub(r"@\S", "", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"\'\w+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\w*\d+\w*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s[^\w\s]\s", "", text)
    return text
