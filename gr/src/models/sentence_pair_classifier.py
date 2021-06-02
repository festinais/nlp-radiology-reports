from argparse import ArgumentParser
from datetime import datetime
from typing import Optional
import torch.nn as nn
import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    glue_compute_metrics
)


class SentencePairClassifier(pl.LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            num_labels: int,
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            eval_splits: Optional[list] = None,
            freeze_bert=False,
            n_classes=2,
            steps_per_epoch=None,
            n_epochs=None,
            **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
        self.dropout = nn.Dropout(p=0.1)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.metric = datasets.load_metric(
            'glue',
            self.hparams.task_name,
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        # Metrics
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self, input_ids, attn_masks, token_type_ids):
        cont_reps, pooler_output = self.model(input_ids, attn_masks, token_type_ids)
        features = self.classifier(self.dropout(pooler_output))
        return features

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        # Option 2
        # We can directly implement the method ".compute()" from the accuracy function
        accuracy = self.train_accuracy.compute()
        # print(f"Train Accuracy: {accuracy}")

        # Save the metric
        self.log('Train_acc_epoch', accuracy, prog_bar=True)

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        return loss

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                    (len(train_loader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
                    // self.hparams.accumulate_grad_batches
                    * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GLUETransformer")
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parent_parser
