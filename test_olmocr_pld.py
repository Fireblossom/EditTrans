"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import argparse
import json
from pathlib import Path
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, LayoutLMv3ImageProcessor, NougatProcessor
import torch
from tqdm import tqdm
import time

from edit_trans.edit_trans_olmocr import EditTransOlmOCR, build_prompt
from nougat.utils.device import move_to_device
from datasets import ClassLabel

from ernie_layout_pytorch.networks import ErnieLayoutTokenizerFast, ErnieLayoutProcessor
from nougat.dataset.feature_processor.ernie_processor import ErnieProcessor
from nougat.dataset.code_doc_dataset import RainbowBankDataset
from ernie_layout_pytorch.networks import ErnieLayoutConfig, set_config_for_extrapolation

def test(args):
    tokenizer_config = torch.load('tokenizer_config.pt')
    tokenizer_config["mask_token"] = "<mask>"
    tokenizer_config["unk_token"] = "<unk>"
    tokenizer_config["pad_token"] = "<pad>"
    tokenizer_config["cls_token"] = "<s>"
    tokenizer_config["sep_token"] = "</s>"
    tokenizer_config["tokenizer_file"] = "tokenizer.json"
    tokenizer = ErnieLayoutTokenizerFast(**tokenizer_config)
    tokenizer.padding_side = 'right'
    tokenizer.only_label_first_subword = False

    image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
    ernie_processor = ErnieLayoutProcessor(image_processor=image_processor, tokenizer=tokenizer)
    data_processor = ErnieProcessor(data_dir=args.data_dir, ernie_processor=ernie_processor, max_length=1024, edit_label=True, test=True)
    test_dataset = RainbowBankDataset(
        data_dir=args.data_dir,
        data_processor=data_processor,
        dataset_name=args.test_dataset_name
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        collate_fn=None,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
        num_workers=6
    )

    ner_labels = ClassLabel(names=['DELETE', 'INSERT_LEFT', 'KEEP', "[DUMMY]"])
    filter_config = ErnieLayoutConfig.from_pretrained('Norm/ERNIE-Layout-Pytorch', output_hidden_states=True)
    filter_config.num_classes = ner_labels.num_classes
    filter_config.use_flash_attn = True
    filter_config.label2id = ner_labels._str2int
    filter_config.id2label = ner_labels._int2str
    filter_config.adapter_pth_name = 'lightning_logs_ernie_edit'
    filter_config.pretrained_model_path = 'Norm/ERNIE-Layout-Pytorch'
    set_config_for_extrapolation(filter_config)

    times = {
        'filter': [],
        'edit': [],
        'olmocr': [],
        'build': [],
        'generation': []
    }

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16)
    model = move_to_device(model)

    edit_model = EditTransOlmOCR(filter_config, model)
    edit_model = move_to_device(edit_model)

    edit_model.eval()

    nougat_processor = NougatProcessor.from_pretrained('facebook/nougat-base')

    tgt_path = Path(args.data_dir)/'mmd_olmocr'
    img_path = Path(args.data_dir)/'images'
    steps_edit = []
    steps_olmocr = []

    for idx, sample in tqdm(enumerate(data_loader), total=len(data_loader)):
        if sample is None:
            continue

        ground_truth = []
        for uid in [Path(n).stem for n in sample['uid']]:
            with open(tgt_path/(uid+'.mmd')) as tgt_mmd:
                ground_truth.append(tgt_mmd.read())

        olmocr_inputs = build_prompt(sample['uid'], processor)
        prompt_len = olmocr_inputs['input_ids'].size(1)
        start_olmocr = time.time()
        
        olmocr_inputs = {k: olmocr_inputs[k].cuda() for k in olmocr_inputs}
        olmocr_inputs['pixel_values'] = olmocr_inputs['pixel_values'].to(dtype=torch.bfloat16)
        outputs_olmocr = model.generate(
            **olmocr_inputs,
            min_length=1,
            max_new_tokens=2048,
            prompt_lookup_num_tokens=3
        )
        end_olmocr = time.time()
        olmocr_time = end_olmocr - start_olmocr
        step_olmocr = outputs_olmocr.size(0) * (outputs_olmocr.size(1)-prompt_len)
        print('baseline:', olmocr_time, step_olmocr)
        times['olmocr'].append(olmocr_time)

    result_path: Path = Path('results/olmocr_pld')/(args.test_dataset_name.split('.')[0])
    result_path.mkdir(parents=True, exist_ok=True)

    with open(result_path/'times.json', 'w') as file:
        json.dump(times, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/rainbow_bank/')
    parser.add_argument("--test_dataset_name", type=str, default='bad_case.txt')
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    args, left_argv = parser.parse_known_args()
    datasets = ['arxiv.txt', 'econ.txt', 'quant_ph.txt']
    for dataset in datasets:
        args.test_dataset_name = dataset
        predictions = test(args)