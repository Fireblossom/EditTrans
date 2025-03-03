from transformers import NougatProcessor, VisionEncoderDecoderModel, LayoutLMv3ImageProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from ernie_layout_pytorch.networks import ErnieLayoutConfig, set_config_for_extrapolation
from ernie_layout_pytorch.networks import exErnieLayoutForTokenClassification, ErnieLayoutTokenizerFast, ErnieLayoutProcessor

from transformers import PreTrainedModel
from transformers.generation.stopping_criteria import StoppingCriteriaList, EosTokenCriteria, MaxLengthCriteria, StoppingCriteria
from transformers.generation.logits_process import LogitsProcessorList
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.generation.configuration_utils import GenerationConfig

import re
import time

import torch
from pytorch_forecasting.utils import padded_stack


class StopTokenCriteria(StoppingCriteria):
    # yet only support stop token with batch size 1
    def __init__(self, stop_strings: list[str], tokenizer: PreTrainedTokenizerBase, device=None):
        self.stop_tokens = tokenizer(stop_strings, return_tensors='pt', padding=True, truncation=True)['input_ids'][:,1:-1].to(device)
        self.stop_tokens_add_space = tokenizer([' '+stop_string for stop_string in stop_strings], return_tensors='pt', padding=True, truncation=True)['input_ids'][:,1:-1].to(device)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.Tensor:
        if input_ids.size(1) < self.stop_tokens.size(1) or input_ids.size(1) < self.stop_tokens_add_space.size(1):
            return False
            
        ret_1 = any([torch.any(input_ids[i, -self.stop_tokens.size(1):] == self.stop_tokens[i,:]) for i in range(input_ids.size(0))])
        ret_2 = any([torch.any(input_ids[i, -self.stop_tokens_add_space.size(1):] == self.stop_tokens_add_space[i,:]) for i in range(input_ids.size(0))])
        return ret_1 or ret_2
    

def find_first_long_word(text):
    # This regex matches a word with more than 2 letters
    match = re.search(r'\b\w{3,}\b', text)
    if match:
        start, end = match.start(), match.end()
        return [text[start:end], text[end:]]
    else:
        return None


def build_edit_seq(logits: torch.Tensor, inputs: dict) -> list[list[str]]:
    batch_size = logits.size(0)
    segment_idxs = inputs['token_in_bboxes'].detach().cpu()
    segment_text = inputs['texts']
    predictions = torch.argmax(logits, dim=-1).detach().cpu()
    edit_seqs = []
    for i in range(batch_size):
        votes = {}
        edit_seq = [None]
        for pred, seg_id in zip(predictions[i,:].tolist(), segment_idxs[i,:].tolist()):
            if seg_id > -1:
                if seg_id not in votes:
                    votes[seg_id] = {0:0, 1:0, 2:0}
                votes[seg_id][pred] += 1

        for seg_id, vote in votes.items():
            label = max(vote, key=vote.get)
            if label == 1: # INSERT
                if edit_seq and edit_seq[-1] is not None:
                    edit_seq.append(None)
                match = find_first_long_word(segment_text[seg_id][i].strip())
                if match and len(match[1]) > 5:
                    edit_tuple = [match[0], match[1]] # stop_string, add_string
                    edit_seq.append(edit_tuple)
            elif label == 0: # DELETE
                pass
            else: # KEEP
                if segment_text[seg_id][i].strip():
                    if edit_seq and edit_seq[-1] is not None:
                        edit_seq[-1][1] += ' ' + segment_text[seg_id][i].strip()
                    else:
                        match = find_first_long_word(segment_text[seg_id][i].strip())
                        if match and len(match[1]) > 5:
                            edit_tuple = [match[0], match[1]] # stop_string, add_string
                            edit_seq.append(edit_tuple)
        if not edit_seq: # empty page?
            edit_seq.append(None)
        elif len(segment_text) >= max(vote.keys()) and edit_seq[-1] is not None:
            edit_seq.append(None) # this page longer than filter prediction, do more insert
        edit_seqs.append(edit_seq)
    return edit_seqs


def sync_batch(input_ids: torch.Tensor, add_next:list[torch.LongTensor]):
    lst = [
            torch.cat([
                input_ids[i,:], #[input_ids[i,:]>2], 
                add_next[i]
            ]) for i in range(input_ids.size(0))
        ]
    ret = padded_stack(lst, side='left')
    return ret


class EditTransNougat(PreTrainedModel):
    def __init__(
            self,
            filter_config: ErnieLayoutConfig, 
        ):
        super().__init__(config=filter_config)
        self.filter_model = exErnieLayoutForTokenClassification.from_pretrained(
            config=filter_config,
            pretrained_model_name_or_path=filter_config.pretrained_model_path,
            ignore_mismatched_sizes=True
        )
        self.filter_model.load_adapter(filter_config.adapter_pth_name)

        self.nougat_model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
        self.processor = NougatProcessor.from_pretrained("facebook/nougat-base")
        self.tokenizer = self.processor.tokenizer

    def get_next(self, edit_seqs):
        stop_strings = []
        add_next = []
        for seq in edit_seqs:
            if not seq: #empty edit seq, generate normally
                add_next.append(torch.LongTensor([]).to(self.device))
            else:
                next_action = seq.pop(0)
                if next_action is None: # NEED INSERT
                    if seq: # has next keep seq
                        stop_strings.append(seq[0][0])
                    add_next.append(torch.LongTensor([]).to(self.device))
                else: # KEEP
                    token_ids = torch.LongTensor(self.tokenizer.encode(next_action[1])[1:-1]).to(self.device)
                    add_next.append(token_ids)
                    if seq: # has next action
                        assert seq.pop(0) == None, 'Next action should be [INSERT]'
                        if seq: # has one more next keep seq
                            stop_strings.append(seq[0][0])
        return stop_strings, add_next

    def generate(
        self, 
        filter_inputs,
        nougat_inputs,
    ):
        generation_config = GenerationConfig(
            _pad_token_tensor = self.tokenizer.pad_token_id,
            output_attentions = False,
            output_hidden_states = False,
            output_scores = False,
            output_loss = False,
            return_dict_in_generate = True,
            max_length = 1024,
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
        end_build = time.time()
        build_time = end_build - start_build

        with torch.no_grad():
            pixel_values = nougat_inputs['pixel_values'].to(self.device, dtype=torch.bfloat16)
            input_ids = torch.zeros([batch_size, 1], dtype=torch.long).to(filter_outputs.logits.device)
            encoder_outputs = BaseModelOutput(hidden_states=self.nougat_model.get_encoder()(pixel_values).last_hidden_state)
            
            # Init inputs to model
            
            edit_time = []
            generation_time = []
            generate_steps = []

            steps = torch.LongTensor([1]*batch_size).to(self.device) # for paper
            start_edit = time.time()
            stop_strings, add_next = self.get_next(edit_seqs)
            input_ids = sync_batch(input_ids, add_next)
            len_before = torch.sum(input_ids>2, dim=-1)
            if not stop_strings:
                stop_strings = None
            end_edit = time.time()
            edit_time.append(end_edit - start_edit)
            
            prepared_stopping_criteria = StoppingCriteriaList()
            prepared_stopping_criteria.append(EosTokenCriteria(self.tokenizer.eos_token_id))
            prepared_stopping_criteria.append(StopTokenCriteria(stop_strings=stop_strings, tokenizer=self.tokenizer, device=self.device))
            prepared_stopping_criteria.append(MaxLengthCriteria(max_length=1024))
            prepared_logits_processor = LogitsProcessorList() # we don't need NoBadWords and ForcedEosToken ATM

            start_edit_generation = time.time()
            outputs = self.nougat_model._sample(
                input_ids=input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=False,
                streamer=None,
                encoder_outputs = encoder_outputs
            )
            end_edit_generation = time.time()
            len_after = torch.sum(outputs.sequences>2, dim=-1)
            steps += len_after-len_before
            cache = outputs.past_key_values
            generation_time.append(end_edit_generation - start_edit_generation)
            generate_steps.append((len_after-len_before).item())

            while any(outputs.sequences[:,-1]>2): # batch not finished generation
                start_edit = time.time()
                stop_strings, add_next = self.get_next(edit_seqs)
                input_ids = sync_batch(outputs.sequences, add_next)
                if input_ids.size(1) > 1022:
                    break # early stop
                len_before = torch.sum(input_ids>2, dim=-1)
                if not stop_strings:
                    stop_strings = None
                    del prepared_stopping_criteria[1]
                else:
                    prepared_stopping_criteria[1] = StopTokenCriteria(stop_strings=stop_strings, tokenizer=self.tokenizer, device=self.device)
                end_edit = time.time()
                edit_time.append(end_edit - start_edit)

                start_edit_generation = time.time()
                # update stop_strings
                outputs = self.nougat_model._sample(
                    input_ids=input_ids,
                    logits_processor=prepared_logits_processor,
                    stopping_criteria=prepared_stopping_criteria,
                    generation_config=generation_config,
                    synced_gpus=False,
                    streamer=None,
                    encoder_outputs = encoder_outputs,
                    past_key_values=cache
                )
                cache = outputs.past_key_values
                len_after = torch.sum(outputs.sequences>2, dim=-1)
                steps += len_after-len_before
                end_edit_generation = time.time()
                generation_time.append(end_edit_generation-start_edit_generation)
                generate_steps.append((len_after-len_before).item())
        return outputs.sequences, steps, filter_time, edit_time, generation_time, generate_steps, build_time