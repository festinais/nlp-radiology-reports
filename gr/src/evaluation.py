from argparse import ArgumentParser
from datetime import datetime
from typing import Optional

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    glue_compute_metrics
)
from pytorch_lightning.loggers import TensorBoardLogger



from src.models.sentence_pair_classifier import SentencePairClassifier
from src.data.data_module import DataModule


def parse_args(args=None):
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = SentencePairClassifier.add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(args)


def main(args):
    pl.seed_everything(args.seed)
    dm = DataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup('fit')
    model = SentencePairClassifier(num_labels=dm.num_labels, eval_splits=dm.eval_splits, **vars(args))
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    return dm, model, trainer