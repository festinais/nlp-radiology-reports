import csv
import glob
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data.data_module import CustomDataset
from datasets import load_metric
from models.sentence_pair_classifier import SentencePairClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import train_bert
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.parameters import get_yaml_parameter
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_data():
    documents = []
    for filename in os.listdir('../data'):
        if filename.endswith('.txt'):
            print(filename)
            with open(os.path.join('../data', filename)) as f:
                content = f.read()
                content = content.replace("\n", "")
                sections = re.split(r'Diagnosenschlüssel:', content)
                documents.append([sections[0], "Diagnosenschlüssel:" + sections[1], '1'])

    with open('../data/data.csv', 'w+') as output:
        writer = csv.writer(output)
        writer.writerow(['section_one', 'section_two', 'label'])
        writer.writerows(documents)

    dataset = load_dataset('csv', data_files='../data/data.csv')

    split = dataset['train'].train_test_split(test_size=0.2, seed=1)  # split the original training data for validation
    train = split['train']
    test = split['test']

    split_val = train.train_test_split(test_size=0.25, seed=1)  # split the original training data for validation
    val = split_val['train']

    df_train = pd.DataFrame(train)
    df_val = pd.DataFrame(val)
    df_test = pd.DataFrame(test)
    return df_train, df_val, df_test


def load_train_val_data(df_train, df_val, df_test):
    # Creating instances of training and validation set
    print("Reading training data...")
    train_set = CustomDataset(df_train, get_yaml_parameter("maxlen"), get_yaml_parameter("bert_model"))

    print("Reading validation data...")
    val_set = CustomDataset(df_val, get_yaml_parameter("maxlen"), get_yaml_parameter("bert_model"))

    print("Reading test data...")
    test_set = CustomDataset(df_test, get_yaml_parameter("maxlen"), get_yaml_parameter("bert_model"))

    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=get_yaml_parameter("bs"), num_workers=5)
    val_loader = DataLoader(val_set, batch_size=get_yaml_parameter("bs"), num_workers=5)
    test_loader = DataLoader(test_set, batch_size=get_yaml_parameter("bs"), num_workers=5)

    return train_loader, val_loader, test_loader


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy()


def test_prediction(net, device, dataloader, with_labels=True, result_file="results/output.txt"):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    net.eval()
    w = open(result_file, 'w')
    probs_all = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, _ in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()
        else:
            for seq, attn_masks, token_type_ids in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()

    w.writelines(str(prob) + '\n' for prob in probs_all)
    w.close()


def evaluate(path_to_output_file, df_test):
    labels_test = df_test['label']  # true labels
    probs_test = pd.read_csv(path_to_output_file, header=None)[0]  # prediction probabilities
    threshold = 0.5  # you can adjust this threshold for your own dataset
    preds_test = (probs_test >= threshold).astype('uint8')  # predicted labels using the above fixed threshold
    metric = load_metric("glue", "mrpc")

    # Compute the accuracy and F1 scores
    score = metric.compute(predictions=preds_test, references=labels_test)
    return score


def main():
    set_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SentencePairClassifier(get_yaml_parameter("bert_model"), freeze_bert=get_yaml_parameter("freeze_bert"))

    if torch.cuda.device_count() > 1:  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(device)

    # get the data
    df_train, df_val, df_test = get_data()
    train_loader, val_loader, test_loader = load_train_val_data(df_train, df_val, df_test)

    criterion = nn.BCEWithLogitsLoss()
    opti = AdamW(net.parameters(), lr=float(get_yaml_parameter("lr")), weight_decay=1e-2)
    num_warmup_steps = 0  # The number of steps for the warmup phase.

    t_total = (len(
        train_loader) // get_yaml_parameter("iters_to_accumulate")) * get_yaml_parameter(
        "epochs")  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=t_total)

    # train the model
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

    # test the model
    path_to_model = glob.glob("models/*.pt")[0]
    if not os.path.exists('results'):
        os.makedirs('result')

    path_to_output_file = get_yaml_parameter('path_to_output_file')
    model = SentencePairClassifier(get_yaml_parameter("bert_model"))
    if torch.cuda.device_count() > 1:  # if multiple GPUs
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    print()
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)

    print("Predicting on test data...")
    test_prediction(net=model, device=device, dataloader=test_loader, with_labels=True,
                    # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                    result_file=path_to_output_file)
    print()
    print("Predictions are available in : {}".format(path_to_output_file))

    # evaluate the model accuracy
    score = evaluate(path_to_output_file, df_test)
    print(score)


if __name__ == "__main__":
    main()
