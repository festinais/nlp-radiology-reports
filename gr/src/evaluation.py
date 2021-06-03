import torch
import torch.nn as nn
import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from data.data_module import CustomDataset
from utils.parameters import get_yaml_parameter
from models.sentence_pair_classifier import SentencePairClassifier
from train import train_bert

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_data():
    dataset = load_dataset('glue', 'mrpc')
    split = dataset['train'].train_test_split(test_size=0.1, seed=1)  # split the original training data for validation
    train = split['train']  # 90 % of the original training data
    val = split['test']  # 10 % of the original training data
    test = dataset[
        'validation']

    # Transform data into pandas dataframes
    df_train = pd.DataFrame(train)
    df_val = pd.DataFrame(val)
    df_test = pd.DataFrame(test)
    return df_train, df_val, df_test


def load_train_val_data(df_train, df_val):
    # Creating instances of training and validation set
    print("Reading training data...")
    train_set = CustomDataset(df_train, get_yaml_parameter("maxlen"), get_yaml_parameter("bert_model"))
    print("Reading validation data...")
    val_set = CustomDataset(df_val, get_yaml_parameter("maxlen"), get_yaml_parameter("bert_model"))
    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=get_yaml_parameter("bs"), num_workers=5)
    val_loader = DataLoader(val_set, batch_size=get_yaml_parameter("bs"), num_workers=5)
    return train_loader, val_loader


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    set_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SentencePairClassifier(get_yaml_parameter("bert_model"), freeze_bert=get_yaml_parameter("freeze_bert"))

    if torch.cuda.device_count() > 1:  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(device)

    df_train, df_val, df_test = get_data()
    train_loader, val_loader = load_train_val_data(df_train, df_val)

    criterion = nn.BCEWithLogitsLoss()
    opti = AdamW(net.parameters(), lr=float(get_yaml_parameter("lr")), weight_decay=1e-2)
    num_warmup_steps = 0  # The number of steps for the warmup phase.
    num_training_steps = get_yaml_parameter("epochs") * len(train_loader)  # The total number of training steps
    t_total = (len(
        train_loader) // get_yaml_parameter("iters_to_accumulate")) * get_yaml_parameter("epochs")  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=t_total)

    train_bert(net,
               criterion,
               opti,
               float(get_yaml_parameter("lr")),
               lr_scheduler,
               train_loader,
               val_loader,
               get_yaml_parameter("epochs"),
               get_yaml_parameter("iters_to_accumulate"),
               device,
               get_yaml_parameter("bert_model"))


if __name__ == "__main__":
    main()
