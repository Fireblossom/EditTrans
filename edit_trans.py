from transformers import NougatProcessor, VisionEncoderDecoderModel, LayoutLMv3ImageProcessor
from src.model.layoutlm_v3.modeling_layoutlmv3 import LayoutLMv3ForTokenClassification
from src.model.layoutlm_v3.configuration_layoutlmv3 import LayoutLMv3Config
from transformers import PreTrainedModel
import re

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
        filter_outputs = self.filter_model(**filter_inputs)
        batch_size = filter_outputs.logits.size(0)
        edit_seqs = build_edit_seq(filter_outputs.logits, filter_inputs)

        # Init inputs to model
        pixel_values = nougat_inputs['pixel_values'].to(self.device, dtype=torch.bfloat16)
        input_ids = torch.zeros([batch_size, 1], dtype=torch.long).to(filter_outputs.logits.device)

        steps = torch.LongTensor([1]*batch_size).to(self.device) # for paper
        stop_strings, add_next = self.get_next(edit_seqs)
        input_ids = sync_batch(input_ids, add_next)
        len_before = torch.sum(input_ids>2, dim=-1)
        if not stop_strings:
            stop_strings = None
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

        while any(input_ids[:,-1]>2): # batch not finished generation
            stop_strings, add_next = self.get_next(edit_seqs)
            input_ids = sync_batch(input_ids, add_next)
            if input_ids.size(1) > 510:
                break # early stop
            len_before = torch.sum(input_ids>2, dim=-1)
            if not stop_strings:
                stop_strings = None
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

        return input_ids, steps
    

if __name__ == "__main__":
    from datasets import ClassLabel

    labels = ClassLabel(names=['[DUMMY]', 'K', 'D', 'I'])
    filter_config = LayoutLMv3Config.from_pretrained('microsoft/layoutlmv3-base', output_hidden_states=True)
    filter_config.pretrained_model_path = 'microsoft/layoutlmv3-base'
    filter_config.adapter_pth_name = 'filter_pth_40000'
    filter_config.num_labels = labels.num_classes
    filter_config.label2id = labels._str2int
    filter_config.id2label = labels._int2str
    filter_config.enable_position_1d = False
    filter_config.device = torch.device('cuda:1')
    
    model = EditTrans(filter_config)