import json
import math
import os
import torch.multiprocessing as mp
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from distutils.util import strtobool
from datasets import ClassLabel
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.argparse import add_argparse_args
from transformers import get_linear_schedule_with_warmup
from src.dataset.augment_processor.aug_for_rnd_move import AugForRndMove
from src.dataset.augment_processor.aug_for_segment_split import AugForSegmentSplit
from src.dataset.code_doc_data_module import CodeDocDataModule
from src.dataset.code_doc_dataset import CodeDocReadingOrderDataset
from src.dataset.feature_processor.filter_pointer_processor import FilterPointerProcessor
from src.model.layoutlm_v3.configuration_layoutlmv3 import LayoutLMv3Config
from src.model.layoutlm_v3.modeling_layoutlmv3 import LayoutLMv3ForTokenClassification
from torchmetrics.classification import MulticlassF1Score
from src.utils.log_utils import create_logger_v2
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, LayoutLMv3ImageProcessor, LayoutLMForTokenClassification
from transformers import LayoutLMConfig


def peft_model_factory(model):
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="lora_only",
        task_type=TaskType.TOKEN_CLS,
        modules_to_save=["classifier"]
    )

    return get_peft_model(model, config)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class LayoutLMv3FilterModelModule(pl.LightningModule):
    def __init__(self,
                 ner_labels: ClassLabel,
                 num_samples: int,
                 learning_rate: float = 2e-5,
                 adam_epsilon: float = 1e-6,
                 warmup_ratio: float = 0.05,
                 **kargs):

        super().__init__()
        self.save_hyperparameters(ignore=['collate_fn', 'tokenizer'])

        # # 定义本地日志
        # if self.global_rank == 0:
        #     self.local_logger = create_logger()

        # 加载自定义模型类
        self.config = LayoutLMv3Config.from_pretrained(self.hparams.pretrained_model_path, output_hidden_states=True)
        self.config.label2id = ner_labels._str2int
        self.config.id2label = ner_labels._int2str
        ##  使用num_hidden_layers层Encoder
        if self.hparams.num_hidden_layers is not None and self.hparams.num_hidden_layers > 0:
            self.config.num_hidden_layers = self.hparams.num_hidden_layers

        self.ner_labels = ner_labels

        self.config.num_labels = ner_labels.num_classes
        self.config.classifier_dropout = self.hparams.dropout
        self.config.enable_position_1d = self.hparams.enable_position_1d

        self.model = LayoutLMv3ForTokenClassification.from_pretrained(config=self.config,
                                                                      pretrained_model_name_or_path=self.hparams.pretrained_model_path,
                                                                      ignore_mismatched_sizes=True)

        self.model = peft_model_factory(self.model)
        print_trainable_parameters(self.model)
        # 设置metric
        self.valid_metric = MulticlassF1Score(num_classes=ner_labels.num_classes, average='micro', ignore_index=0, zero_division=1.)

        if self.global_rank == 0:
            self.local_logger = create_logger_v2(log_dir=self.hparams.save_model_dir)
            self.local_logger.info(self.hparams)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        steps = batch_idx

        # 这里在训练时打日志，由 log_every_n_steps 控制频率
        if self.global_rank == 0 and self.local_rank == 0 and (steps + 1) % self.trainer.log_every_n_steps == 0:
            lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
            self.local_logger.info(
                f"Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"Steps: {steps}, "
                f"Learning Rate {lr_scheduler.get_last_lr()[-1]:.7f}, "
                f"Train Loss: {loss:.5f}"
            )

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs.loss, outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        predictions, labels = predictions.detach().cpu(), batch["labels"].detach().cpu()
        self.valid_metric(predictions, labels)
        return val_loss

    def validation_epoch_end(self, validation_step_outputs):

        metric_results = self.valid_metric.compute()

        val_loss = torch.stack(validation_step_outputs).mean()
        val_micro_f1 = metric_results

        self.log("val_micro_f1", val_micro_f1, prog_bar=True, on_epoch=True)

        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(
                f"**Validation** , Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"GlobalSteps: {self.global_step}, "
                f"val_loss: {val_loss:.5f}, "
                f"val_micro_f1: {val_micro_f1:.5f}, "
            )
        # *** 这个一定需要，不然会重复累积 *** #
        self.valid_metric.reset()

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs.loss, outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        predictions, labels = predictions.detach().cpu().numpy(), batch["labels"].detach().cpu().numpy()
        self.valid_metric(predictions, labels)

        return val_loss

    def test_epoch_end(self, outputs):

        metric_results = self.valid_metric.compute()

        val_loss = torch.stack(outputs).mean()
        val_micro_f1 = metric_results

        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(
                f"**Test** , "
                f"test_loss: {val_loss:.5f}, "
                f"test_micro_f1: {val_micro_f1:.5f}, "
            )
        # *** 这个一定需要，不然会重复累积 *** #
        self.valid_metric.reset()

    # 用于模型测试
    def predict_step(self, batch, batch_idx):
        # this calls forward
        inputs = {}
        for k in ['input_ids', 'bbox', 'attention_mask', 'labels', 'position_ids']:
            inputs[k] = batch[k]
        outputs = self(**inputs)
        logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1)

        uids = batch['uid']
        attention_mask = batch['attention_mask'].detach().cpu().numpy()
        predict_ids = predictions.detach().cpu().numpy()
        token2word_info = batch['token2word_info']

        '''
        predictions是token级别的预测结果，需要解析出 word 和 entity级别的结果
        输入：
        - ner labels
        - token2word map
        输出：
        - entities，与输入格式一致
        - 详情
        '''

        ner_labels = self.ner_labels
        predict_results = []

        # 注意第0位是CLS
        for uid, masks, predict_idx, t2ws in zip(uids, attention_mask, predict_ids, token2word_info):
            # batch中的一条数据
            # 获取word级别label

            entities_map = dict()

            t2ws = json.loads(t2ws)

            tag, ent = None, ''

            for idx, mask, predict_id in zip(range(len(masks)), masks, predict_idx):
                if idx == 0:
                    continue
                if mask == 0:
                    break
                if idx >= len(t2ws):
                    break

                label = ner_labels.int2str(int(predict_id))
                words = t2ws[idx]['word_texts']
                if not tag is None: # 目前存在一个解码中的实体
                    if label == f'I-{tag}': # 继续解码当前实体
                        ent += ''.join(words)
                        continue
                    else: # 停止解码当前实体
                        # 保存当前实体
                        if tag not in entities_map:
                            entities_map[tag] = []
                        entities_map[tag].append(ent)
                # 确定是否开启了一个新的实体的解码
                if label.startswith('B-'):
                    tag, ent = label.split('-')[1], ''.join(words)
                else:
                    tag, ent = None, ''
            # 解码最后一个实体
            if not tag is None:
                if tag not in entities_map:
                    entities_map[tag] = []
                entities_map[tag].append(ent)
            # entities_map的keys排序 每个key下的实体列表也排序
            entities_map_new = dict()
            for k in sorted(entities_map):
                entities_map_new[k] = sorted(entities_map[k])
            predict_results.append([uid, entities_map_new])

        return predict_results

    def configure_callbacks(self):
        call_backs = []

        call_backs.append(LearningRateMonitor(logging_interval='step'))
        call_backs.append(EarlyStopping(monitor="val_word_f", mode="max", patience=self.hparams.patience))
        call_backs.append(
            ModelCheckpoint(monitor="val_word_f",
                            mode="max",
                            every_n_epochs=self.hparams.every_n_epochs,
                            filename='codedoc-{epoch}-{step}-{val_word_f:.5f}-{val_micro_f1:.5f}'
                            )
        )

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
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
                                      eps=self.hparams.adam_epsilon)
        num_warmup_steps = int(self.total_steps * self.hparams.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def setup(self, stage=None):
        if stage != "fit":
            return
        num_samples = self.hparams.num_samples
        batch_size = self.hparams.batch_size

        steps = math.ceil(num_samples / batch_size / max(1, self.trainer.devices))
        # Calculate total steps
        ab_steps = int(steps / self.trainer.accumulate_grad_batches)
        # self.total_steps = int(records_count // ab_size * self.trainer.max_epochs)
        self.total_steps = int(ab_steps * self.trainer.max_epochs)
        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(f"- num_samples is: {num_samples}")
            self.local_logger.info(f"- max_epochs is: {self.trainer.max_epochs}")
            self.local_logger.info(f"- total_steps is: {self.total_steps}")
            self.local_logger.info(f"- batch size (1 gpu) is: {batch_size}")
            self.local_logger.info(f"- devices(gpu) num is: {max(1, self.trainer.devices)}")
            self.local_logger.info(f"- accumulate_grad_batches is: {self.trainer.accumulate_grad_batches}")

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        return add_argparse_args(cls, parent_parser, **kwargs)


class LightningRunner:
    def __init__(self, args):
        self.args = args

    def run(self):
        mp.set_start_method('spawn')
        # 设置随机种子
        pl.seed_everything(self.args.seed)

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.args.pretrained_model_path)

        ner_labels = ClassLabel(names=['[DUMMY]', 'K', 'D', 'I'])
        '''
        这里不同的任务使用不同的Processor
        '''
        augment_processors = []
        if self.args.enable_aug:
            augment_processors.append(AugForSegmentSplit(prob=0.2))
            augment_processors.append(AugForRndMove(prob=0.95))

        image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
        data_processor_train = FilterPointerProcessor(tokenizer=tokenizer,
                                                      image_processor=image_processor,
                                                      box_level=self.args.box_level,
                                                      max_text_length=self.args.max_length,
                                                      norm_bbox_height=self.args.norm_bbox_height,
                                                      norm_bbox_width=self.args.norm_bbox_width,
                                                      augment_processors=augment_processors,
                                                      use_image=self.args.use_image,
                                                      data_dir=self.args.data_dir
                                                      )

        data_processor = FilterPointerProcessor(tokenizer=tokenizer,
                                                image_processor=image_processor,
                                                box_level=self.args.box_level,
                                                max_text_length=self.args.max_length,
                                                norm_bbox_height=self.args.norm_bbox_height,
                                                norm_bbox_width=self.args.norm_bbox_width,
                                                use_image=self.args.use_image,
                                                data_dir=self.args.data_dir
                                                )

        # 定义数据
        train_dataset, valid_dataset, test_dataset = None, None, None
        num_samples = 1
        if self.args.train_dataset_name and self.args.do_train:
            train_dataset = CodeDocReadingOrderDataset(data_dir=self.args.data_dir,
                                                       data_processor=data_processor_train,
                                                       dataset_name=self.args.train_dataset_name
                                                       )
            num_samples = len(train_dataset)
        if self.args.valid_dataset_name and self.args.do_train:
            valid_dataset = CodeDocReadingOrderDataset(data_dir=self.args.data_dir,
                                                       data_processor=data_processor,
                                                       dataset_name=self.args.valid_dataset_name
                                                       )
        if self.args.test_dataset_name and (self.args.do_test or self.args.do_predict):
            test_dataset = CodeDocReadingOrderDataset(data_dir=self.args.data_dir,
                                                      data_processor=data_processor,
                                                      dataset_name=self.args.test_dataset_name
                                                      )
            
        data_module = CodeDocDataModule(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            **vars(self.args)
        )

        # 定义模型

        model = LayoutLMv3FilterModelModule(
            ner_labels=ner_labels,
            num_samples=num_samples,
            **vars(self.args)
        )
        # 定义日志 本地tensorboard
        logger = TensorBoardLogger(save_dir=self.args.save_model_dir, name='')
        # 定义trainer
        trainer = Trainer.from_argparse_args(self.args,
                                             weights_save_path=args.save_model_dir,
                                             logger=logger,
                                             enable_progress_bar=False,
                                             plugins=[LightningEnvironment()])

        if self.args.do_train:
            trainer.fit(model, data_module)

        if self.args.do_test:
            trainer.test(model, data_module, ckpt_path='best' if self.args.ckpt_path is None else self.args.ckpt_path)

        if self.args.do_predict:
            predictions = trainer.predict(model, data_module,
                                          ckpt_path='best' if self.args.ckpt_path is None else self.args.ckpt_path)

            fw = open(self.args.predict_result_file, 'w')
            for prediction_batch in predictions:
                for uid, prediction_map in prediction_batch:
                    fw.write('\t'.join([uid, json.dumps(prediction_map, ensure_ascii=False)]))
                    fw.write('\n')
            fw.close()
            
        best_model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        best_model.model.save_pretrained('filter_pth_40000')


def main(args):
    gpus = args.gpus
    if gpus > 1 and args.strategy is None:
        args.strategy = 'ddp'
    runner = LightningRunner(args)
    runner.run()


if __name__ == '__main__':
    # 添加conflict_handler，防止和trainer参数冲突
    parser = ArgumentParser(conflict_handler='resolve')
    parser = Trainer.add_argparse_args(parser)
    parser = CodeDocDataModule.add_argparse_args(parser)

    # Data Hyperparameters
    parser.add_argument('--data_dir', default='rainbow_bank', type=str)
    parser.add_argument('--train_dataset_name', default='train.jsonl', type=str)
    parser.add_argument('--valid_dataset_name', default='test.jsonl', type=str)
    parser.add_argument('--test_dataset_name', default='toy.jsonl', type=str)
    parser.add_argument('--box_level', default='segment', type=str, help='word or segment')
    parser.add_argument('--enable_position_1d', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='是否使用v3的global 1d编码',
                        default=False)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--norm_bbox_height', default=1000, type=int)
    parser.add_argument('--norm_bbox_width', default=1000, type=int)
    parser.add_argument('--use_image', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='是否使用图像模态',
                        default=True)

    parser.add_argument('--shuffle', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='是否shuffle',
                        default=True)

    # Model Hyperparameters
    parser.add_argument('--pretrained_model_path', default='microsoft/layoutlmv3-base',
                        type=str)
    parser.add_argument('--num_hidden_layers', default=-1, type=int, help='默认使用12层Bert')
    parser.add_argument('--dropout', default=0.1, type=float)

    # Basic Training Control

    parser.add_argument('--do_train', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do train',
                        default=True)
    parser.add_argument('--do_test', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do test',
                        default=False)
    parser.add_argument('--do_predict', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do test',
                        default=False)
    parser.add_argument('--predict_result_file', default=None, type=str)
    parser.add_argument('--enable_aug', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='enable aug',
                        default=False)

    parser.add_argument('--precision', default=16, type=int, )
    parser.add_argument('--num_nodes', default=1, type=int, )
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--strategy', default=None, type=str)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--max_steps', default=40000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--val_test_batch_size', default=-1, type=int)
    parser.add_argument('--preprocess_workers', default=32, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--patience', default=50, type=int)

    parser.add_argument('--save_model_dir', default='/nfs/home/duan/tpp/lightning_logs_40000', type=str)
    parser.add_argument('--ckpt_path', default=None, type=str)
    parser.add_argument('--log_every_n_steps', default=1, type=int)
    parser.add_argument('--val_check_interval', default=0.25, type=float)  # int时多少个steps跑验证集,float 按照比例算
    parser.add_argument('--every_n_epochs', default=1, type=int)
    parser.add_argument('--keep_checkpoint_max', default=10, type=int)
    parser.add_argument('--deploy_path', default='', type=str)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--detect_anomaly', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='是否开启detect',
                        default=False)
    args = parser.parse_args()

    main(args)
