import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomDataset(Dataset):

    def __init__(self, data, maxlen=128, with_labels=True, bert_model='albert-base-v2'):

        self.data = data  # pandas dataframe
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data[['section_one']].iloc[[index]])
        sent2 = str(self.data[['section_two']].iloc[[index]])
        label = self.data[['label']].iloc[[index]]
        return sent1, sent2, label
