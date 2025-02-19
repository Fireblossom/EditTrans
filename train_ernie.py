import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import json
import math
import torch.multiprocessing as mp
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from datasets import ClassLabel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from transformers import get_linear_schedule_with_warmup
from nougat.dataset.augment_processor.aug_for_rnd_move import AugForRndMove
from nougat.dataset.augment_processor.aug_for_segment_split import AugForSegmentSplit
from nougat.dataset.code_doc_data_module import CodeDocDataModule
from nougat.dataset.code_doc_dataset import RainbowBankDataset
from nougat.dataset.feature_processor.ernie_processor import ErnieProcessor as ErniePLProcessor

from torchmetrics.classification import MulticlassF1Score, BinaryF1Score
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, LayoutLMv3ImageProcessor

from ernie_layout_pytorch.networks import ErnieLayoutConfig, set_config_for_extrapolation
from ernie_layout_pytorch.networks import exErnieLayoutForTokenClassification, ErnieLayoutTokenizerFast, ErnieLayoutProcessor
from utils import compute_box_predictions
from nougat.utils.log_utils import create_logger_v2

torch.set_float32_matmul_precision('medium')
def peft_model_factory(model):
    config = LoraConfig(
        r=256,
        lora_alpha=256,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="lora_only",
        task_type=TaskType.TOKEN_CLS,
        modules_to_save=["classifier"]
    )
    return get_peft_model(model, config)


def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")


class ErnieLayoutFilterModelModule(pl.LightningModule):
    def __init__(self, ner_labels: ClassLabel, num_samples: int, learning_rate: float = 2e-5, **kargs):
        super().__init__()
        self.save_hyperparameters()

        self.config = ErnieLayoutConfig.from_pretrained(self.hparams.pretrained_model_path, output_hidden_states=True)
        self.config.num_classes = ner_labels.num_classes
        self.config.use_flash_attn = True
        self.config.label2id = ner_labels._str2int
        self.config.id2label = ner_labels._int2str
        self.config.classifier_dropout = self.hparams.dropout

        set_config_for_extrapolation(self.config)
        self.model = exErnieLayoutForTokenClassification.from_pretrained(
            config=self.config,
            pretrained_model_name_or_path=self.hparams.pretrained_model_path,
            ignore_mismatched_sizes=True
        )
        self.model = peft_model_factory(self.model)
        print_trainable_parameters(self.model)

        self.valid_metric = (BinaryF1Score(ignore_index=-100) if self.config.num_classes < 3 else 
                             MulticlassF1Score(self.config.num_classes, average='micro', ignore_index=-100))
        self.valid_metric.to(self.device)

        self.valid_metric_segment = (BinaryF1Score(ignore_index=-100) if self.config.num_classes < 3 else 
                             MulticlassF1Score(self.config.num_classes, average='micro', ignore_index=-100))
        self.valid_metric_segment.to(self.device)

        if self.global_rank == 0:
            self.local_logger = create_logger_v2(log_dir=self.hparams.save_model_dir)
            self.local_logger.info(self.hparams)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        batch['batch_idx'] = batch_idx
        torch.save(batch, 'debug.pt')
        outputs = self(**batch)
        loss = outputs.loss
        steps = batch_idx

        # 这里在训练时打日志，由 log_every_n_steps 控制频率
        if self.global_rank == 0 and self.local_rank == 0 and (steps + 1) % self.trainer.log_every_n_steps == 0:
            self.local_logger.info(
                f"Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"Steps: {steps}, "
                f"Learning Rate {self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]:.7f}, "
                f"Train Loss: {loss:.5f}"
            )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs.loss, outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        self.valid_metric(predictions, batch["labels"].detach())
        return val_loss
    
    def on_validation_epoch_end(self):
        metric_results = self.valid_metric.compute()
        metric_results_seg = self.valid_metric_segment.compute()

        val_micro_f1 = metric_results
        val_micro_f1_seg = metric_results_seg

        self.log("val_micro_f1", val_micro_f1, prog_bar=True, on_epoch=True)
        self.log("val_micro_f1_seg", val_micro_f1_seg, prog_bar=True, on_epoch=True)

        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(
                f"**Validation** , Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"GlobalSteps: {self.global_step}, "
                f"val_micro_f1: {val_micro_f1:.5f}, "
                f"val_micro_f1_seg: {val_micro_f1_seg:.5f}"
            )
        # *** 这个一定需要，不然会重复累积 *** #
        self.valid_metric.reset()
        self.valid_metric_segment.reset()
    
    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs.loss, outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        predictions_seg, labels_seg, count_token_seg = compute_box_predictions(predictions, batch.token_in_bboxes, self.config.num_labels, batch["labels"])
        labels_seg[count_token_seg < 4] = -100
        self.valid_metric_segment(predictions_seg.detach(), labels_seg.detach())
        predictions, labels = predictions.detach(), batch["labels"].detach()
        self.valid_metric(predictions, labels)
        # torch.save([predictions_seg, labels_seg, batch.token_in_bboxes, predictions, labels], 'debug.pt')
        return val_loss

    def on_test_epoch_end(self):
        metric_results = self.valid_metric.compute()
        metric_results_seg = self.valid_metric_segment.compute()

        val_micro_f1 = metric_results
        val_micro_f1_seg = metric_results_seg

        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(
                f"**Test** , "
                f"test_micro_f1_token_level: {val_micro_f1:.5f}, "
                f"test_micro_f1_segment_level: {val_micro_f1_seg:.5f}"
            )
        # *** 这个一定需要，不然会重复累积 *** #
        self.valid_metric.reset()
        self.valid_metric_segment.reset()

    def configure_optimizers(self):
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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                                      eps=1e-6)
        num_warmup_steps = int(self.total_steps * self.hparams.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        return [
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(monitor="val_micro_f1", mode="max", patience=5),
            ModelCheckpoint(monitor="val_micro_f1", mode="max", save_top_k=1, filename='codedoc-{epoch}-{val_micro_f1:.5f}')
        ]
    
    def setup(self, stage=None):
        if stage != "fit":
            return
        num_samples = self.hparams.num_samples
        batch_size = self.hparams.batch_size

        steps = math.ceil(num_samples / batch_size)
        # Calculate total steps
        ab_steps = int(steps / self.trainer.accumulate_grad_batches)
        # self.total_steps = int(records_count // ab_size * self.trainer.max_epochs)
        self.total_steps = int(ab_steps * self.trainer.max_epochs)
        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(f"- num_samples is: {num_samples}")
            self.local_logger.info(f"- max_epochs is: {self.trainer.max_epochs}")
            self.local_logger.info(f"- total_steps is: {self.total_steps}")
            self.local_logger.info(f"- batch size (1 gpu) is: {batch_size}")
            self.local_logger.info(f"- accumulate_grad_batches is: {self.trainer.accumulate_grad_batches}")


class LightningRunner:
    def __init__(self, args):
        self.args = args

    def run(self):
        pl.seed_everything(self.args.seed)

        tokenizer_config = torch.load('tokenizer_config.pt')
        tokenizer_config["mask_token"] = "<mask>"
        tokenizer_config["unk_token"] = "<unk>"
        tokenizer_config["pad_token"] = "<pad>"
        tokenizer_config["cls_token"] = "<s>"
        tokenizer_config["sep_token"] = "</s>"
        tokenizer_config["tokenizer_file"] = "tokenizer.json"
        tokenizer = ErnieLayoutTokenizerFast(**tokenizer_config)

        ner_labels = ClassLabel(names=['DELETE', 'INSERT_LEFT', 'KEEP', "[DUMMY]"] if self.args.edit_label else ['DELETE', 'KEEP'])

        tokenizer.padding_side = 'right'
        tokenizer.only_label_first_subword = False
        '''
        这里不同的任务使用不同的Processor
        '''
        augment_processors = []
        if self.args.enable_aug:
            augment_processors.append(AugForSegmentSplit(prob=0.2))
            augment_processors.append(AugForRndMove(prob=0.95))

        image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
        processor = ErnieLayoutProcessor(image_processor=image_processor, tokenizer=tokenizer)

        data_processor_train = ErniePLProcessor(data_dir=self.args.data_dir, ernie_processor=processor, max_length=self.args.max_length, edit_label=self.args.edit_label)
        data_processor = ErniePLProcessor(data_dir=self.args.data_dir, ernie_processor=processor, max_length=self.args.max_length, edit_label=self.args.edit_label)

        # 定义数据
        train_dataset, valid_dataset, test_dataset = None, None, None
        num_samples = 1
        if self.args.train_dataset_name and self.args.do_train:
            train_dataset = RainbowBankDataset(data_dir=self.args.data_dir,
                                                       data_processor=data_processor_train,
                                                       dataset_name=self.args.train_dataset_name
                                                       )
        if self.args.valid_dataset_name and self.args.do_train:
            valid_dataset = RainbowBankDataset(data_dir=self.args.data_dir,
                                                       data_processor=data_processor,
                                                       dataset_name=self.args.valid_dataset_name
                                                       )
        if self.args.test_dataset_name and (self.args.do_test or self.args.do_predict):
            test_dataset = RainbowBankDataset(data_dir=self.args.data_dir,
                                                      data_processor=data_processor,
                                                      dataset_name=self.args.test_dataset_name
                                                      )
            
        data_module = CodeDocDataModule(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            **vars(self.args)
        )
        if self.args.do_train:
            num_samples = len(data_module.train_dataloader().dataset)
        model = ErnieLayoutFilterModelModule(ner_labels=ner_labels, num_samples=num_samples, **vars(self.args))
        logger = TensorBoardLogger(save_dir=self.args.save_model_dir, name='')
        trainer = Trainer(
            max_epochs=self.args.max_epochs,
            default_root_dir=self.args.save_model_dir,
            logger=logger,
            accelerator='auto',
            strategy="auto",
            callbacks=model.configure_callbacks()
        )
        if self.args.do_train:
            trainer.fit(model, data_module)
            model.model.save_pretrained(self.args.save_model_dir)

        if self.args.do_test:
            trainer.test(model, data_module, ckpt_path='best' if self.args.ckpt_path is None else self.args.ckpt_path)
        


def main(args):
    runner = LightningRunner(args)
    runner.run()


if __name__ == '__main__':
    from distutils.util import strtobool
    parser = ArgumentParser()
    parser.add_argument('--pretrained_model_path', type=str, default='Norm/ERNIE-Layout-Pytorch')
    parser.add_argument('--save_model_dir', type=str, default='./lightning_logs_ernie_edit')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--edit_label', action='store_true', default=True)

    parser.add_argument('--learning_rate', type=float, default=4e-5)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--precision', default=16, type=int, )
    parser.add_argument('--num_nodes', default=1, type=int, )
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--strategy', default=None, type=str)
    parser.add_argument('--max_steps', default=40000, type=int)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--val_test_batch_size', default=48, type=int)
    parser.add_argument('--preprocess_workers', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--log_every_n_steps', default=10, type=int)

    parser.add_argument('--data_dir', default='/raid/duan/cd58hofa/rainbow_bank_edit/', type=str)
    parser.add_argument('--train_dataset_name', default='train.txt', type=str)
    parser.add_argument('--valid_dataset_name', default='test.txt', type=str)
    parser.add_argument('--test_dataset_name', default='test.txt', type=str)
    parser.add_argument('--box_level', default='segment', type=str, help='word or segment')
    parser.add_argument('--max_length', default=1024, type=int)
    parser.add_argument('--norm_bbox_height', default=1000, type=int)
    parser.add_argument('--norm_bbox_width', default=1000, type=int)
    parser.add_argument('--use_image', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='是否使用图像模态',
                        default=True)
    parser.add_argument('--do_train', type=lambda x: bool(strtobool(x)), nargs='?', const=False, help='do train',
                        default=True)
    parser.add_argument('--do_test', type=lambda x: bool(strtobool(x)), nargs='?', const=False, help='do test',
                        default=False)
    parser.add_argument('--do_predict', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do test',
                        default=False)
    parser.add_argument('--enable_aug', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='enable aug',
                        default=False)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--ckpt_path', default=None, type=str)
    args = parser.parse_args()
    main(args)
