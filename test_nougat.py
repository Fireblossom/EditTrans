"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import argparse
import json
from multiprocessing import Pool
from collections import defaultdict
from pathlib import Path
from transformers import NougatProcessor, VisionEncoderDecoderModel, LayoutLMv3ImageProcessor
import numpy as np
import torch
from tqdm import tqdm
import time

from edit_trans.edit_trans_nougat import EditTransNougat
from nougat.metrics import compute_metrics
from nougat.utils.device import move_to_device
from datasets import ClassLabel
from PIL import Image
import os
from ernie_layout_pytorch.networks import ErnieLayoutTokenizerFast, ErnieLayoutProcessor
from nougat.dataset.feature_processor.ernie_processor import ErnieProcessor
from nougat.dataset.code_doc_dataset import RainbowBankDataset
from ernie_layout_pytorch.networks import ErnieLayoutConfig, set_config_for_extrapolation


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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

    edit_model = EditTransNougat(filter_config)
    edit_model = move_to_device(edit_model)

    edit_model.eval()

    metrics = defaultdict(list)
    metrics_nougat = defaultdict(list)

    times = {
        'filter': [],
        'edit': [],
        'nougat': [],
        'build': [],
        'generation': []

    }

    processor = NougatProcessor.from_pretrained("facebook/nougat-base")
    model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
    model = move_to_device(model)

    tgt_path = Path(args.data_dir)/'mmd_nougat'
    img_path = Path(args.data_dir)/'images'
    steps_edit = []
    steps_nougat = []
    for idx, sample in tqdm(enumerate(data_loader), total=len(data_loader)):
        if sample is None:
            continue

        # image_tensors, decoder_input_ids, _ = sample
        # if image_tensors is None:
        #     return

        ground_truth = []
        for uid in [Path(n).stem for n in sample['uid']]:
            with open(tgt_path/(uid+'.mmd')) as tgt_mmd:
                ground_truth.append(tgt_mmd.read())

        # ground_truth = edit_model.tokenizer.batch_decode(
        #     decoder_input_ids, skip_special_tokens=True
        # )
        
        nougat_inputs = edit_model.processor([Image.open(img_path/img) for img in sample['uid']],
                                                    return_tensors="pt")
        start_nougat = time.time()
        outputs_nougat = model.generate(
            nougat_inputs['pixel_values'].to(model.device, dtype=torch.bfloat16),
            min_length=1,
            max_new_tokens=1024,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
        )
        end_nougat = time.time()
        nougat_time = end_nougat - start_nougat
        step_nougat = outputs_nougat.size(0) * outputs_nougat.size(1)
        print('baseline:', nougat_time, step_nougat)
        times['nougat'].append(nougat_time)

        outputs, steps, filter_time, edit_time, generation_time, generation_steps, build_time = edit_model.generate(
            filter_inputs=sample,
            nougat_inputs=nougat_inputs,
        )
        outputs_text = edit_model.processor.batch_decode(
            outputs, skip_special_tokens=True
        )
        outputs_text = edit_model.processor.post_process_generation(outputs_text, fix_markdown=True)
        #text, math, table = split_text(outputs_text)

        ground_truth_text = [ground_truth[i][:len(text)] for i, text in enumerate(outputs_text)] # keep lengths same for too long generation
        ground_truth_text = edit_model.processor.post_process_generation(ground_truth_text, fix_markdown=True)
        #text_gt, math_gt, table_gt = split_text(ground_truth_text)

        print('editTrans:',filter_time, build_time, sum(edit_time), sum(generation_time), sum(generation_steps))
        times['filter'].append(filter_time)
        times['build'].append(build_time)
        times['edit'].append(sum(edit_time))
        times['generation'].append(sum(generation_time))

        outputs_text_nougat = edit_model.processor.batch_decode(
            outputs_nougat, skip_special_tokens=True
        )
        outputs_text_nougat = edit_model.processor.post_process_generation(outputs_text_nougat, fix_markdown=True)
        #text_nougat, math_nougat, table_nougat = split_text(outputs_text_nougat)

        with Pool(args.batch_size) as p:
            _metrics = p.starmap(compute_metrics, iterable=zip(outputs_text, ground_truth_text))
            for m in _metrics:
                for key, value in m.items():
                    metrics[key].append(value)
            print({key: sum(values) / len(values) for key, values in metrics.items()})
        
        with Pool(args.batch_size) as p:
            _metrics = p.starmap(compute_metrics, iterable=zip(outputs_text_nougat, ground_truth_text))
            for m in _metrics:
                for key, value in m.items():
                    metrics_nougat[key].append(value)
            print({key: sum(values) / len(values) for key, values in metrics_nougat.items()})

        steps_edit.append(torch.sum(steps).item())
        steps_nougat.append(outputs_nougat.size(0) * outputs_nougat.size(1))
        print(sum(steps_edit)/(idx+1), sum(steps_nougat)/(idx+1))

    result_path = Path('results/nougat')/(args.test_dataset_name.split('.')[0])
    result_path.mkdir(parents=True, exist_ok=True)
    with open(result_path/'score_edit.json', 'w') as file:
        json.dump(metrics, file, indent=2)
    with open(result_path/'score_nougat.json', 'w') as file:
        json.dump(metrics_nougat, file, indent=2)

    with open(result_path/'steps.json', 'w') as file:
        json.dump({
            'edit': steps_edit,
            'nougat': steps_nougat
        }, file, indent=2)
    with open(result_path/'times.json', 'w') as file:
        json.dump(times, file, indent=2)
    scores = {}
    for metric, vals in metrics.items():
        scores[f"{metric}_accuracies"] = vals
        scores[f"{metric}_accuracy"] = np.mean(vals)
    try:
        print(
            f"Total number of samples: {len(vals)}, Edit Distance (ED) based accuracy score: {scores['edit_dist_accuracy']}, BLEU score: {scores['bleu_accuracy']}, METEOR score: {scores['meteor_accuracy']}"
        )
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/rainbow_bank/')
    parser.add_argument("--test_dataset_name", type=str, default='arxiv.txt')
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    args, left_argv = parser.parse_known_args()

    predictions = test(args)
