from transformers import Kosmos2_5Processor, Kosmos2_5ForConditionalGeneration, PreTrainedModel
from ernie_layout_pytorch.networks import ErnieLayoutConfig
from ernie_layout_pytorch.networks import exErnieLayoutForTokenClassification
from transformers.generation.stopping_criteria import StoppingCriteriaList, EosTokenCriteria, MaxLengthCriteria
from transformers.generation.logits_process import LogitsProcessorList
from torch.nn import functional as F
from transformers.generation.configuration_utils import GenerationConfig
import time
import torch
from edit_trans.edit_utils import NGramMatchStopCriteria, build_edit_seq, sync_batch

class EditTransKosmos(PreTrainedModel):
    def __init__(
            self,
            filter_config: ErnieLayoutConfig, 
            kosmos_model: Kosmos2_5ForConditionalGeneration,
        ):
        super().__init__(config=filter_config)
        self.filter_model = exErnieLayoutForTokenClassification.from_pretrained(
            config=filter_config,
            pretrained_model_name_or_path=filter_config.pretrained_model_path,
            ignore_mismatched_sizes=True
        )
        self.filter_model.load_adapter(filter_config.adapter_pth_name)

        self.kosmos_model = kosmos_model # Kosmos2_5ForConditionalGeneration.from_pretrained("microsoft/kosmos-2.5")
        self.processor = Kosmos2_5Processor.from_pretrained("microsoft/kosmos-2.5")
        self.tokenizer = self.processor.tokenizer

    def tokenize_edit_seqs(self, edit_seqs: list[list[str]]) -> list[list[torch.LongTensor|None]]:
        tokenized_seqs = []
        for seq in edit_seqs:
            tokenized_seq = []
            for action in seq:
                if action is None:
                    tokenized_seq.append(None)
                else:
                    ids = torch.LongTensor(self.tokenizer.encode(action)).to(self.device)
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
        kosmos_inputs,
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
        edit_seqs = self.tokenize_edit_seqs(build_edit_seq(filter_outputs.logits, filter_inputs))
        end_build = time.time()
        build_time = end_build - start_build

        with torch.no_grad():
            vision_model_output = self.kosmos_model.vision_model(flattened_patches=kosmos_inputs['flattened_patches'])
            image_embeds = F.normalize(vision_model_output[0], dim=-1)
            image_embeds, projection_attentions = self.kosmos_model.image_to_text_projection(image_embeds)

            edit_time = []
            generation_time = []
            generate_steps = []

            steps = torch.LongTensor([0]*batch_size).to(self.device)
            start_edit = time.time()
            stop_strings, add_next = self.get_next(edit_seqs)
            input_ids = sync_batch(kosmos_inputs['input_ids'], add_next, [0]*batch_size)
            len_before = kosmos_inputs['input_ids'].size(1)
            end_edit = time.time()
            edit_time.append(end_edit - start_edit)
            
            prepared_stopping_criteria = StoppingCriteriaList()
            prepared_stopping_criteria.append(EosTokenCriteria(self.tokenizer.eos_token_id))
            prepared_stopping_criteria.append(NGramMatchStopCriteria(stop_strings))
            prompt_len = kosmos_inputs['input_ids'].size(1)
            prepared_stopping_criteria.append(MaxLengthCriteria(max_length=1024+prompt_len))
            prepared_logits_processor = LogitsProcessorList() # we don't need NoBadWords and ForcedEosToken ATM
            kosmos_inputs['input_ids'] = input_ids
            kosmos_inputs['image_embeds'] = image_embeds
            del kosmos_inputs['flattened_patches']

            start_edit_generation = time.time()
            outputs = self.kosmos_model.text_model._sample(
                **kosmos_inputs,
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
                if input_ids.size(1)-prompt_len > 1024:
                    break # early stop
                len_before = input_ids.size(1)
                prepared_stopping_criteria[1] = NGramMatchStopCriteria(stop_strings)
                end_edit = time.time()
                edit_time.append(end_edit - start_edit)

                start_edit_generation = time.time()
                # update stop_strings
                outputs = self.kosmos_model.text_model._sample(
                    input_ids=input_ids,
                    past_key_values=cache,
                    # attention_mask=kosmos_inputs['attention_mask'],
                    # image_embeds_position_mask=kosmos_inputs['image_embeds_position_mask'],
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