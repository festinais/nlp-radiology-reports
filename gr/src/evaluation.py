import glob
import os
import random
import numpy as np
import pandas as pd
import torch
from data.data_module import CustomDataset
from datasets import load_metric
from models.sentence_pair_classifier import SentencePairClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import train_bert
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.parameters import get_yaml_parameter
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification

# SimCLR
from simclr.simclr import SimCLR
from simclr.modules import NT_Xent

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_data():
    # train = pd.read_csv('gr/data/data_train.csv')
    # val = pd.read_csv('gr/data/data_val.csv')
    # test = pd.read_csv('gr/data/data_test.csv')

    # split = dataset['train'].train_test_split(test_size=0.2, seed=1)  # split the original training data for validation
    # train = split['train']
    # val = split['test']
    df = load_dataset('csv', data_files='gr/data_1/mrpc_data.csv')

    split = df['train'].train_test_split(test_size=0.2, seed=1)  # split the original training data for validation
    train = split['train']

    split_new = train.train_test_split(test_size=0.1, seed=1)
    train = split_new['train']
    val = split_new['test']

    test = split['test']

    df_train = pd.DataFrame(train)
    df_val = pd.DataFrame(val)
    # df_test = pd.read_csv("gr/data/data_no_dup_test.csv", nrows=122)
    df_test = pd.DataFrame(test)

    print('{0} {1} length'.format(df_train.shape, 'train'))
    print('{0} {1} length'.format(df_val.shape, 'validation'))
    print('{0} {1} length'.format(df_test.shape, 'test'))
    return df_train, df_val, df_test


def collate_fn(batch):
    tokenizer = AutoTokenizer.from_pretrained(get_yaml_parameter("bert_model"))
    maxlen = get_yaml_parameter("maxlen")

    token_ids, attn_masks, token_type_ids, labels = [], [], [], []

    for index, tuple in enumerate(batch):
        sent1 = tuple[0]
        sent2 = tuple[1]

        labels.append(torch.tensor(tuple[2].values))
        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = tokenizer(sent1, sent2,
                                 padding='max_length',  # Pad to max_length
                                 truncation=True,  # Truncate to max_length
                                 max_length=maxlen,
                                 return_tensors='pt')  # Return torch.Tensor objects

        token_ids.append(encoded_pair['input_ids'].squeeze(0))  # tensor of token ids
        attn_masks.append(encoded_pair['attention_mask'].squeeze(
            0))  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids.append(encoded_pair['token_type_ids'].squeeze(
            0))  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        # negative sampling
        if index == len(batch) - 1:
            index = -1
            sent3 = batch[index + 1][1]
        else:
            sent3 = batch[index + 1][1]
        label = torch.tensor(0)

        labels.append(label)
        encoded_pair = tokenizer(sent1, sent3,
                                 padding='max_length',  # Pad to max_length
                                 truncation=True,  # Truncate to max_length
                                 max_length=maxlen,
                                 return_tensors='pt')  # Return torch.Tensor objects

        token_ids.append(encoded_pair['input_ids'].squeeze(0))  # tensor of token ids
        attn_masks.append(encoded_pair['attention_mask'].squeeze(0))
        token_type_ids.append(encoded_pair['token_type_ids'].squeeze(0))

    return torch.stack(token_ids), torch.stack(attn_masks), torch.stack(token_type_ids), torch.LongTensor(labels)


def load_train_val_data(df_train, df_val, df_test):
    # Creating instances of training and validation set
    print("Reading training data...")
    train_set = CustomDataset(df_train)

    print("Reading validation data...")
    val_set = CustomDataset(df_val)

    print("Reading test data...")
    test_set = CustomDataset(df_test)

    # Creating instances of training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=get_yaml_parameter("bs"), drop_last=True)
    val_loader = DataLoader(val_set, batch_size=get_yaml_parameter("bs"), drop_last=True)
    test_loader = DataLoader(test_set, batch_size=get_yaml_parameter("bs"), drop_last=True)

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


def test_prediction(net, device, dataloader, criterion, with_labels=True, result_file="results/output.txt"):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    net.eval()
    w = open(result_file, 'w')

    # accuracies
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    top_k_accuracies = []

    tokenizer = AutoTokenizer.from_pretrained(get_yaml_parameter("bert_model"))
    count = 0
    mean_acc = 0
    for it, (section_ones, section_two, labels) in enumerate(tqdm(dataloader)):
        encoded_pairs_1 = tokenizer(list(section_ones),
                                    padding='max_length',  # Pad to max_length
                                    truncation=True,  # Truncate to max_length
                                    max_length=128,
                                    return_tensors='pt')  # Return torch.Tensor objects

        encoded_pairs_2 = tokenizer(list(section_two),
                                    padding='max_length',  # Pad to max_length
                                    truncation=True,  # Truncate to max_length
                                    max_length=128,
                                    return_tensors='pt')  # Return torch.Tensor objects

        input_ids_1 = encoded_pairs_1['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks_1 = encoded_pairs_1['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids_1 = encoded_pairs_1['token_type_ids'].squeeze(
            0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        input_ids_2 = encoded_pairs_2['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks_2 = encoded_pairs_2['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids_2 = encoded_pairs_2['token_type_ids'].squeeze(
            0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        # Converting to cuda tensors
        input_ids_1, attn_masks_1, token_type_ids_1, labels = input_ids_1.to(device), attn_masks_1.to(
            device), token_type_ids_1.to(device), labels.to(device)
        input_ids_2, attn_masks_2, token_type_ids_2 = input_ids_2.to(device), attn_masks_2.to(
            device), token_type_ids_2.to(device)

        # Enables autocasting for the forward pass (model + loss)
        # with autocast():
            # Obtaining the logits from the model
        h_i, h_j, z_i, z_j = net(input_ids_1, attn_masks_1, token_type_ids_1, input_ids_2, attn_masks_2,
                                 token_type_ids_2)
        # Computing loss
        loss, acc, logits, labels = criterion(h_i, h_j)
        mean_acc += acc
        count += 1

        metric_acc.add_batch(predictions=logits, references=labels)
        metric_f1.add_batch(predictions=logits, references=labels)

        # compute the top k predicted classes, per pixel:
        scores, indices = torch.topk(logits, 3)
        # you now have k predictions per pixel, and you want that one of them will match the true labels y:
        correct = labels[indices]
        top_k_accuracies.append(correct)

        # top_k_accuracies.append(top_k_accuracy_score(logits, labels, k=3))

    final_score_acc = metric_acc.compute()
    final_score_f1 = metric_f1.compute(average=None)
    top_3_acc = sum(top_k_accuracies) / len(top_k_accuracies)
    return final_score_acc, final_score_f1, top_3_acc


def accuracy(output, target, topk=(3,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
    # device = torch.device("cpu")
    net = SentencePairClassifier(get_yaml_parameter("bert_model"), freeze_bert=get_yaml_parameter("freeze_bert"))

    bert_layer = AutoModel.from_pretrained(get_yaml_parameter("bert_model"), return_dict=False)
    configs = bert_layer.config

    # initialize model
    net = SimCLR(net, 64, configs.hidden_size)

    # if torch.cuda.device_count() > 1:  # if multiple GPUs
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     net = nn.DataParallel(net)

    # device = torch.device("cpu")
    net.to(device)

    # get the data
    df_train, df_val, df_test = get_data()
    train_loader, val_loader, test_loader = load_train_val_data(df_train, df_val, df_test)

    criterion = NT_Xent(get_yaml_parameter("bs"), 0.5, 1)
    # criterion = nn.BCEWithLogitsLoss()

    param_optimizer = list(net.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)

    # opti = AdamW(net.parameters(), lr=float(get_yaml_parameter("lr")), weight_decay=1e-2)
    # num_warmup_steps = 0  # The number of steps for the warmup phase.

    t_total = (len(train_loader)) * get_yaml_parameter(
        "epochs")  # Necessary to take into account Gradient accumulation
    lr_scheduler = 0

    # train the model
    train_bert(net,
               criterion,
               optimizer,
               float(get_yaml_parameter("lr")),
               lr_scheduler,
               train_loader,
               val_loader,
               get_yaml_parameter("epochs"),
               get_yaml_parameter("iters_to_accumulate"),
               device,
               get_yaml_parameter("bert_model"))


def evaluate_main():
    # test the model
    device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df_train, df_val, df_test = get_data()

    train_loader, val_loader, test_loader = load_train_val_data(df_train, df_val, df_test)

    path_to_model = glob.glob("models/*.pt")[0]
    if not os.path.exists('results'):
        os.makedirs('results')

    path_to_output_file = get_yaml_parameter('path_to_output_file')
    # model = SentencePairClassifier(get_yaml_parameter("bert_model"), freeze_bert=get_yaml_parameter("freeze_bert"))
    #
    # bert_layer = AutoModel.from_pretrained(get_yaml_parameter("bert_model"), return_dict=False)
    # configs = bert_layer.config
    # model = SimCLR(model, 64, configs.hidden_size)

    # if torch.cuda.device_count() > 1:  # if multiple GPUs
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    print("Loading the mode.")
    # print("Loading the weights of the model...")
    # model.load_state_dict(torch.load(path_to_model))
    # model.load_state_dict(torch.load(path_to_model, map_location=device))

    model = torch.load(path_to_model)

    model.eval()
    model.to(device)

    print("Predicting on test data...")
    criterion = NT_Xent(get_yaml_parameter("bs"), 0.5, 1)
    score_acc, score_f1, top_3 = test_prediction(net=model, device=device, dataloader=test_loader, criterion=criterion,
                                                 with_labels=True,
                                                 # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                                                 result_file=path_to_output_file)

    # evaluate the model accuracy
    # score = evaluate(path_to_output_file, df_test)
    print(score_acc)
    print(score_f1)
    print("top_3 accuracy", top_3)


if __name__ == "__main__":
    # main()
    evaluate_main()
