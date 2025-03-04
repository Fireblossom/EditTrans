from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, PreTrainedModel
from ernie_layout_pytorch.networks import ErnieLayoutConfig
from ernie_layout_pytorch.networks import exErnieLayoutForTokenClassification
from transformers.generation.stopping_criteria import StoppingCriteriaList, EosTokenCriteria, MaxLengthCriteria
from transformers.generation.logits_process import LogitsProcessorList
from torch.nn import functional as F
from transformers.generation.configuration_utils import GenerationConfig
import time
import torch
from edit_trans.edit_utils import NGramMatchStopCriteria, build_edit_seq, sync_batch
from PIL import Image
import base64
from io import BytesIO
from pathlib import Path
from olmocr.prompts import build_finetuning_prompt


class EditTransOlmOCR(PreTrainedModel):
    def __init__(
            self,
            filter_config: ErnieLayoutConfig, 
            olmocr_model: Qwen2VLForConditionalGeneration,
        ):
        super().__init__(config=filter_config)
        self.filter_model = exErnieLayoutForTokenClassification.from_pretrained(
            config=filter_config,
            pretrained_model_name_or_path=filter_config.pretrained_model_path,
            ignore_mismatched_sizes=True
        )
        self.filter_model.load_adapter(filter_config.adapter_pth_name)

        self.olmocr_model = olmocr_model
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.tokenizer = self.processor.tokenizer
        self.add_top_string = '{"primary_language":"en","is_rotation_valid":true,"rotation_correction":0,"is_table":false,"is_diagram":false,"natural_text":"'

    def tokenize_edit_seqs(self, edit_seqs: list[list[str]]) -> list[list[torch.LongTensor|None]]:
        tokenized_seqs = []
        for seq in edit_seqs:
            tokenized_seq = []
            for action in seq:
                if action is None:
                    tokenized_seq.append(None)
                else:
                    ids = torch.LongTensor(self.tokenizer.encode(action)[1:-1]).to(self.device)
                    if ids.size(0) > 5:
                        tokenized_seq.append(ids) # don't insert very short text
                    else:
                        if tokenized_seq and tokenized_seq[-1] is None:
                            tokenized_seq.pop() # remove the last None and skip this insert
            tokenized_seqs.append(tokenized_seq)
        return tokenized_seqs

    def get_next(self, edit_seqs):
        stop_strings = []
        add_next = []
        for seq in edit_seqs:
            if not seq: #empty edit seq, generate normally
                stop_strings.append(False)
                add_next.append(torch.LongTensor([]).to(self.device))
            else:
                next_action = seq.pop(0)
                if next_action is None: # NEED INSERT
                    if seq: # has next keep seq
                        stop_strings.append(seq[0])
                    else:
                        stop_strings.append(False)
                    add_next.append(torch.LongTensor([]).to(self.device))
                else: # KEEP
                    add_next.append(next_action)
                    if seq: # has next action
                        assert seq.pop(0) == None, 'Next action should be [INSERT_LEFT]'
                        if seq: # has one more next keep seq
                            stop_strings.append(seq[0])
                        else:
                            stop_strings.append(False)
                    else:
                        stop_strings.append(False)
        return stop_strings, add_next

    def generate(
        self, 
        filter_inputs,
        olmocr_inputs,
    ):
        generation_config = GenerationConfig(
            _pad_token_tensor = self.tokenizer.pad_token_id,
            output_attentions = False,
            output_hidden_states = False,
            output_scores = False,
            output_loss = False,
            return_dict_in_generate = True,
            do_sample = False,
        )
        for k, v in filter_inputs.items():
            if isinstance(v, torch.LongTensor):
                filter_inputs[k] = v.to(self.device)
            elif isinstance(v, torch.FloatTensor):
                filter_inputs[k] = v.to(self.device, dtype=torch.bfloat16)
        start_filter = time.time()
        filter_outputs = self.filter_model(**filter_inputs)
        end_filter = time.time()
        filter_time = end_filter - start_filter
        batch_size = filter_outputs.logits.size(0)
        start_build = time.time()
        edit_seqs = build_edit_seq(filter_outputs.logits, filter_inputs)
        edit_seqs = [[self.add_top_string]+edit_seq for edit_seq in edit_seqs]
        edit_seqs = self.tokenize_edit_seqs(edit_seqs)
        end_build = time.time()
        build_time = end_build - start_build

        with torch.no_grad():
            prompt_len = olmocr_inputs['input_ids'].size(1)
            edit_time = []
            generation_time = []
            generate_steps = []

            steps = torch.LongTensor([0]*batch_size).to(self.device)
            start_edit = time.time()
            stop_strings, add_next = self.get_next(edit_seqs)
            input_ids = sync_batch(olmocr_inputs['input_ids'], add_next, [0]*batch_size)
            len_before = input_ids.size(1)
            olmocr_inputs['attention_mask'] = torch.ones([batch_size,len_before]).to(device=input_ids.device, dtype=torch.long)
            end_edit = time.time()
            edit_time.append(end_edit - start_edit)
            
            prepared_stopping_criteria = StoppingCriteriaList()
            prepared_stopping_criteria.append(EosTokenCriteria(self.tokenizer.eos_token_id))
            prepared_stopping_criteria.append(NGramMatchStopCriteria(stop_strings))
            prepared_stopping_criteria.append(MaxLengthCriteria(max_length=2048+prompt_len))
            prepared_logits_processor = LogitsProcessorList() # we don't need NoBadWords and ForcedEosToken ATM
            olmocr_inputs['input_ids'] = input_ids

            start_edit_generation = time.time()
            outputs = self.olmocr_model._sample(
                **olmocr_inputs,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=False,
                streamer=None,
            )
            end_edit_generation = time.time()
            len_after = outputs.sequences.size(1)
            steps += len_after-len_before
            cache = outputs.past_key_values
            generation_time.append(end_edit_generation - start_edit_generation)
            generate_steps.append(len_after-len_before)

            while all(outputs.sequences[:,-1]!=self.tokenizer.eos_token_id): # batch not finished generation
                start_edit = time.time()
                stop_strings, add_next = self.get_next(edit_seqs)
                input_ids = sync_batch(outputs.sequences, add_next, prepared_stopping_criteria[1].get_matched_add_position())
                if input_ids.size(1)-prompt_len > 2048:
                    break # early stop
                len_before = input_ids.size(1)
                prepared_stopping_criteria[1] = NGramMatchStopCriteria(stop_strings)
                end_edit = time.time()
                edit_time.append(end_edit - start_edit)

                start_edit_generation = time.time()
                # update stop_strings
                outputs = self.olmocr_model._sample(
                    input_ids=input_ids,
                    past_key_values=cache,
                    # attention_mask=olmocr_inputs['attention_mask'],
                    # image_embeds_position_mask=olmocr_inputs['image_embeds_position_mask'],
                    # the image info. is already encoded into the past keys/values
                    logits_processor=prepared_logits_processor,
                    stopping_criteria=prepared_stopping_criteria,
                    generation_config=generation_config,
                    synced_gpus=False,
                    streamer=None,
                )
                cache = outputs.past_key_values
                len_after = outputs.sequences.size(1)
                steps += len_after-len_before
                end_edit_generation = time.time()
                generation_time.append(end_edit_generation-start_edit_generation)
                generate_steps.append(len_after-len_before)
        return outputs.sequences, steps, filter_time, edit_time, generation_time, generate_steps, build_time
    

def read_and_resize_image(path:str | Path, target_longest_image_dim=1024) -> Image:
    img = Image.open(path)
    width, height = img.size
    if width > height:
        new_width = target_longest_image_dim
        new_height = int(height * target_longest_image_dim / width)
    else: 
        new_height = target_longest_image_dim
        new_width = int(width * target_longest_image_dim / height)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="png")
    image_base64 = base64.b64encode(buffered.getvalue())
    return img, image_base64


def build_prompt(uids, processor):
    anchor_text_path = Path('data/rainbow_bank/anchor_text_olmocr')
    image_path = Path('data/rainbow_bank/images')
    imgs, texts = [], []
    for uid in uids:
        img, image_base64 = read_and_resize_image(image_path/uid)
        uid = uid.replace('.png', '')
        with open(anchor_text_path/uid) as f:
            anchor_text = f.read()
        prompt = build_finetuning_prompt(anchor_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs.append(img)
        texts.append(text)

    inputs = processor(
        text=texts,
        images=imgs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


