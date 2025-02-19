from transformers import NougatProcessor, VisionEncoderDecoderModel, LayoutLMv3ImageProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from src.model.layoutlm_v3.modeling_layoutlmv3 import LayoutLMv3ForTokenClassification
from src.model.layoutlm_v3.configuration_layoutlmv3 import LayoutLMv3Config
from transformers import PreTrainedModel
from transformers.generation.stopping_criteria import StopStringCriteria, StoppingCriteriaList
from torch.nn import functional as F
import re
import time

import torch
from pytorch_forecasting.utils import padded_stack


def find_first_long_word(text):
    # This regex matches a word with more than 3 letters
    match = re.search(r'\b\w{4,}\b', text)
    if match:
        start, end = match.start(), match.end()
        return [text[start:end], text[end:]]
    else:
        return None


def build_edit_seq(logits: torch.Tensor, inputs: dict) -> list[list[str]]:
    batch_size = logits.size(0)
    segment_idxs = inputs['segment_idxs'].detach().cpu()
    segment_text = [eval(s) for s in inputs['segment_text']]
    predictions = torch.argmax(logits, dim=-1).detach().cpu()
    edit_seqs = []
    for i in range(batch_size):
        votes = {}
        edit_seq = [None]
        for pred, seg_id in zip(predictions[i,:].tolist(), segment_idxs[i,:].tolist()):
            if seg_id not in votes:
                votes[seg_id] = {1:0, 2:0, 3:0}
            if seg_id == 0 or pred == 0:
                continue
            votes[seg_id][pred] += 1

        for seg_id, vote in votes.items():
            label = max(vote, key=vote.get)
            if label == 3: # INSERT
                if edit_seq and edit_seq[-1] is not None:
                    edit_seq.append(None)
                match = find_first_long_word(segment_text[i][seg_id].strip())
                if match and len(match[1]) > 5:
                    edit_tuple = [match[0], match[1]] # stop_string, add_string
                    edit_seq.append(edit_tuple)
            elif label == 2: # DELETE
                pass
            else: # KEEP
                if segment_text[i][seg_id].strip():
                    if edit_seq and edit_seq[-1] is not None:
                        edit_seq[-1][1] += ' ' + segment_text[i][seg_id].strip()
                    else:
                        match = find_first_long_word(segment_text[i][seg_id].strip())
                        if match and len(match[1]) > 5:
                            edit_tuple = [match[0], match[1]] # stop_string, add_string
                            edit_seq.append(edit_tuple)
        if not edit_seq: # empty page?
            edit_seq.append(None)
        elif len(segment_text[i]) >= max(vote.keys()) and edit_seq[-1] is not None:
            edit_seq.append(None) # this page longer than filter prediction, do more insert
        edit_seqs.append(edit_seq)
    return edit_seqs


def sync_batch(input_ids: torch.Tensor, add_next:list[torch.LongTensor]):
    device = input_ids.device
    # add_next = [x.to(device) for x in add_next]
    return padded_stack([
            torch.cat([
                torch.LongTensor([0]).to(device),
                input_ids[i,:][input_ids[i,:]>2], 
                add_next[i]
            ]) for i in range(input_ids.size(0))
        ], side='left'
    )


class PerRowStopStringCriteria(StopStringCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.Tensor:
        #return super().__call__(input_ids, scores, **kwargs)
        assert self.num_stop_strings+1 == input_ids.size(0)

        self.embedding_vec = self.embedding_vec.to(input_ids.device)
        self.target_lens = self.target_lens.to(input_ids.device)
        # The maximum length we need to consider is 1 token per character. Note that input_ids can also be
        # *shorter* than the global max, and the code below should be ready for that
        input_ids = input_ids[:, -self.maximum_token_len :]

        # Flip input_ids because we're only matching strings at the end of the generated sequence
        flipped_ids = torch.flip(input_ids, (1,))

        # Size of the vector of positions a single token can match
        max_valid_positions = self.max_valid_positions

        # The embedding vec contains the valid positions, end_lengths and total lengths for each token
        embedded = F.embedding(flipped_ids, self.embedding_vec)

        # Now we split the embedding vector. valid_positions is the positions in the stop string the token can fit
        valid_positions = embedded[:, 1:, : max_valid_positions * self.num_stop_strings].unflatten(
            -1, (self.num_stop_strings, -1)
        )
        # end_lengths is the number of characters from the string, counting from the end, that the token
        # contains. It can have multiple values if the same token can overlap different end lengths
        end_lengths = embedded[:, :1, max_valid_positions * self.num_stop_strings : -1].unflatten(
            -1, (self.num_stop_strings, -1)
        )
        # Lengths is the total length of each token. Unlike the others, it always has a single value
        lengths = embedded[:, 1:, None, -1:]  # Insert a dummy dimension for stop_strings even though lengths are const

        # Concatenate lengths onto each possible end_lengths value
        lengths = lengths.expand((-1, -1, end_lengths.shape[-2], end_lengths.shape[-1]))
        lengths_with_ends = torch.cat([end_lengths, lengths], dim=1)

        # cumsum() to get the number of matched characters in the stop string after each token
        cumsum = lengths_with_ends.cumsum(dim=1)  # B x maximum_token_len x num_stop_strings x max_valid_end_lens

        # The calculation above assumes that all tokens are in valid positions. Now we mask the ones that are not.
        # First, tokens match the start of the string if they have a positive value in the end_lengths vector
        initial_match = end_lengths > 0

        # Tokens continue the string if the cumsum() so far is one of the valid positions for that token
        # Note that we're actually tracking one cumsum() for for each possible end_length
        later_match = torch.any(cumsum[:, :-1, :, None] == valid_positions[:, :, :, :, None], axis=-2)

        # The match vector is a boolean vector that indicates which positions have valid tokens
        match = torch.cat([initial_match, later_match], dim=1)

        # Once a single position does not match, all positions following that position are masked
        mask = (~match).cumsum(dim=1, dtype=torch.int32)
        mask = mask == 0

        # The string is matched if we reached a cumsum equal to or greater than the length of the string
        # before hitting the mask
        string_matches = torch.amax(cumsum * mask, dim=(1, -1)) >= self.target_lens[None, :]

        # We return a per-sample vector that is True if the n-th stop string is matched for that sample
        return torch.cat([string_matches.diag(), torch.BoolTensor([False]).to(string_matches.device)])


class EditTrans(PreTrainedModel):
    base_model_prefix = "editocr"
    def __init__(
            self,
            filter_config: LayoutLMv3Config, 
        ):
        super().__init__(config=filter_config)
        self.filter_config = filter_config

        self.filter_model = LayoutLMv3ForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=filter_config.pretrained_model_path,
            config=filter_config,
            ignore_mismatched_sizes=True,
        )
        self.filter_model.load_adapter(filter_config.adapter_pth_name)
        self.filter_model.to(filter_config.device)
        self.filter_image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)

        self.nougat_model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
        self.processor = NougatProcessor.from_pretrained("facebook/nougat-base")
        self.tokenizer = self.processor.tokenizer
        # self.device = self.filter_model.device
        # self.nougat_model.to(self.device)
        self.inner_batch_limit=256

        # assert self.filter_model.device == self.nougat_model.device, 'models should be on the same device'

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
    
    def build_inner_batch(self, edit_seqs):
        batch = []
        for i, seq in enumerate(edit_seqs):
            stop_string, add_next = self.get_next([seq])
            while stop_string != [] or add_next[0].size(0) != 0:
                add_next_last_5 = add_next[0]
                batch.append((i, stop_string, add_next_last_5))
                stop_string, add_next = self.get_next([seq])
        return batch

    def inference_batch(
        self,
        filter_inputs,
        nougat_inputs,
    ):
        for k, v in filter_inputs.items():
            if isinstance(v, torch.LongTensor):
                filter_inputs[k] = v.to(self.device)
            elif isinstance(v, torch.FloatTensor):
                filter_inputs[k] = v.to(self.device, dtype=torch.bfloat16)
        filter_outputs = self.filter_model(**filter_inputs)
        batch_size = filter_outputs.logits.size(0)
        edit_seqs = build_edit_seq(filter_outputs.logits, filter_inputs)

        # Init inputs to model
        pixel_values = nougat_inputs['pixel_values'].to(self.device, dtype=torch.bfloat16)

        edit_seq_batches = self.build_inner_batch(edit_seqs)
        pixel_values_batch = torch.stack([pixel_values[i] for i, stop_string, add_next in edit_seq_batches])
        input_ids = padded_stack([add_next for i, stop_string, add_next in edit_seq_batches], side='left')
        stop_strings = [stop_string[0] for i, stop_string, add_next in edit_seq_batches if stop_string]
        stop_criterias = StoppingCriteriaList()
        stop_criterias.append(PerRowStopStringCriteria(self.tokenizer, stop_strings=stop_strings))

        input_ids = self.nougat_model.generate(
            pixel_values_batch,
            input_ids=input_ids,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            tokenizer=self.tokenizer,
            stopping_criteria=stop_criterias,
            max_length=512
        )

        pass


    def inference(
        self, 
        filter_inputs,
        nougat_inputs,
    ):
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

        # Init inputs to model
        start_edit = time.time()
        pixel_values = nougat_inputs['pixel_values'].to(self.device, dtype=torch.bfloat16)
        input_ids = torch.zeros([batch_size, 1], dtype=torch.long).to(filter_outputs.logits.device)
        
        edit_time = []
        generation_time = []
        generate_steps = []

        steps = torch.LongTensor([1]*batch_size).to(self.device) # for paper
        stop_strings, add_next = self.get_next(edit_seqs)
        input_ids = sync_batch(input_ids, add_next)
        len_before = torch.sum(input_ids>2, dim=-1)
        if not stop_strings:
            stop_strings = None
        end_edit = time.time()
        edit_time.append(end_edit - start_edit)
        
        start_edit_generation = time.time()
        input_ids = self.nougat_model.generate(
            pixel_values,
            input_ids=input_ids,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            stop_strings=stop_strings,
            tokenizer=self.tokenizer,
            max_length=512
        )
        len_after = torch.sum(input_ids>2, dim=-1)
        steps += len_after-len_before
        end_edit_generation = time.time()
        generation_time.append(end_edit_generation - start_edit_generation)
        generate_steps.append((len_after-len_before).item())

        while any(input_ids[:,-1]>2): # batch not finished generation
            start_edit = time.time()
            stop_strings, add_next = self.get_next(edit_seqs)
            input_ids = sync_batch(input_ids, add_next)
            if input_ids.size(1) > 510:
                break # early stop
            len_before = torch.sum(input_ids>2, dim=-1)
            if not stop_strings:
                stop_strings = None
            end_edit = time.time()
            edit_time.append(end_edit - start_edit)

            start_edit_generation = time.time()
            input_ids = self.nougat_model.generate(
                pixel_values,
                input_ids=input_ids,
                bad_words_ids=[[self.tokenizer.unk_token_id]],
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
                max_length=512
            )
            len_after = torch.sum(input_ids>2, dim=-1)
            steps += len_after-len_before
            end_edit_generation = time.time()
            generation_time.append(end_edit_generation-start_edit_generation)
            generate_steps.append((len_after-len_before).item())

        return input_ids, steps, filter_time, edit_time, generation_time, generate_steps, build_time
    

if __name__ == "__main__":
    from datasets import ClassLabel

    labels = ClassLabel(names=['[DUMMY]', 'K', 'D', 'I'])
    filter_config = LayoutLMv3Config.from_pretrained('microsoft/layoutlmv3-base', output_hidden_states=True)
    filter_config.pretrained_model_path = 'microsoft/layoutlmv3-base'
    filter_config.adapter_pth_name = 'filter_pth'
    filter_config.num_labels = labels.num_classes
    filter_config.label2id = labels._str2int
    filter_config.id2label = labels._int2str
    filter_config.enable_position_1d = False
    filter_config.device = torch.device('cuda:0')
    
    model = EditTrans(filter_config)