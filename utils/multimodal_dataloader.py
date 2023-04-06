"""Datasets & DataLoaders
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

PyTorch provides two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset that allow you to use
pre-loaded datasets as well as your own data. Dataset stores the samples and their corresponding labels, and DataLoader
wraps an iterable around the Dataset to enable easy access to the samples.

inputs required by...
audio model
- audio wav
- audio mask
- labels

text model
- text ids  torch.LongTensor of the text
- text mask torch.LongTensor of the attention mask of the text
- labels torch.tensor of the target labels (0 where the text does not have that label, 1 otherwise)
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import transformers


class TextDataset(Dataset):
    """Text dataset for text embeddings."""

    def __init__(self, df: pd.DataFrame, tokenizer, target_encoder, max_len):
        """
        Args:
        :param df: dataframe of utterances and emotion label
        :param tokenizer: tokenizer for creating embedding of utterances
        :param target_encoder: encoding of text labels to integers
        :param max_len: maximum length of embedding
        """
        self.df = df
        self.text = df["text"].values
        self.target = df["target"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_encoder = target_encoder

    def __len__(self):
        """This method means that len(dataset) returns the size of the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Supports the indexing such that dataset[i] can be used to get the ith sample."""
        texts = self.text[idx]
        tokenized = self.tokenizer.encode_plus(text=texts,  # list of strings of utterances
                                               truncation=True,  # truncate to max_length
                                               add_special_tokens=True,
                                               # whether or not to encode the sequences with the special tokens relative to their model
                                               max_length=self.max_len,  # used in padding and trunctation
                                               padding="max_length")  # pad to max_length
        # tokenizer returns a dictionary with all the arguments necessary for its corresponding model to work properly
        # tokenizer takes care of splitting the sequence into tokens available in the tokenizer vocabulary - tokens are either words or subwords
        ids = tokenized[
            "input_ids"]  # list of token ids to be fed to a model. input_ids are token indices, numerical representations of tokens building the sequences that will be used as input by the model
        mask = tokenized[
            "attention_mask"]  # a binary tensor indicating the position of the padded indices so that the model does not attend to them
        targets = self.target_encoder.transform([[self.target[idx]]]).toarray().reshape(
            -1)  # encode the text target labels

        return {
            "ids": torch.LongTensor(ids),
            "mask": torch.LongTensor(mask),
            "targets": torch.tensor(targets, dtype=torch.float32)
        }


"""DataLoader
Using a for loop to iterate over our data means we lose a lot of features such as batching and shuffling the data and
loading the data in parallel. DataLoader is an iterator which provides all these features.

dataset = TextDataset(df, tokenizer, target_encoder, max_len)
dataset_batched = DataLoader(dataset, shuffle=True, batch_size=16)
"""


class AudioDataset(Dataset):
    """Audio dataset for wav2vec2.

    input: audio filepath
    output: Float values of input raw speech waveform (input_values)
    """

    def __init__(self, df: pd.DataFrame, target_encoder, preprocessor: transformers.Wav2Vec2Processor):
        """
        Args:
            df: dataframe of audio filepath (.wav) and emotion label
            target_encoder: encoding of text labels to integers e.g. instance of OneHotEncoder
            preprocessor: wav2vec2 preprocessor object
        """
        self.df = df
        self.filepath = df["filepath"].values
        self.target = df["target"].values
        self.target_encoder = target_encoder
        self.preprocessor = preprocessor

    def __len__(self):
        """This method means that len(dataset) returns the size of the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Supports the indexing such that dataset[i] can be used to get the ith sample.
        See colab notebook https://colab.research.google.com/drive/13ItwffzW-R-vnaHWfugbERoErQsFdCNs#scrollTo=MmO4Q8LCCGQy
        useful links: https://huggingface.co/docs/transformers/v4.27.2/en/glossary#attention-mask
        https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2Processor

        - take in audio filepath
        - get audio and sr
        - convert to single channel (suppress both channels into one)
            - only apply when needed
        - use Wav2Vec2Processor (look at padding and max_length)
            - Speechbrain padded at training stage
        - return what wav2vec2 needs
        """
        pass
