from transformers.generation.stopping_criteria import StoppingCriteria
from pytorch_forecasting.utils import padded_stack
import torch


class NGramMatchStopCriteria(StoppingCriteria):
    # yet only support stop token with batch size 1
    def __init__(self, stop_string: list[torch.LongTensor], ngram_size: int = 3):
        self.stop_string = stop_string
        self.ngram_size = ngram_size
        self.match_position = False
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.Tensor:
        # TODO: support batch size > 1
        # TODO: skip the current stop_string if it's not valid

        if input_ids.size(0) > 1:
            raise NotImplementedError('StopLastNGramCriteria only support batch size 1 ATM')
        
        if self.stop_string[0] is False or input_ids.size(1) < self.ngram_size:
            return False # last insert or not enough tokens
        
        # inspired by PromptLookupCandidateGenerator

        # Create sliding windows of size ngram_size
        windows = self.stop_string[0].unfold(dimension=0, size=self.ngram_size, step=1)

        # Convert ngram to a tensor for comparison
        ngram_tensor = input_ids[0, -self.ngram_size:]

        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=1)

        # Get the indices of matches
        match_indices = matches.nonzero(as_tuple=True)[0]

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            self.match_position = idx + self.ngram_size
            return True
        return False

    def get_matched_add_position(self):
        return [self.match_position,]


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
                match = segment_text[seg_id][i].strip()
                if match and len(match) > 5:
                    edit_seq.append(' ' + match)
            elif label == 0: # DELETE
                pass
            else: # KEEP
                if segment_text[seg_id][i].rstrip():
                    if edit_seq and edit_seq[-1] is not None:
                        edit_seq[-1] += ' ' + segment_text[seg_id][i].lstrip()
                    else:
                        match = segment_text[seg_id][i].strip()
                        if match and len(match) > 5:
                            edit_seq.append(' ' + match)
        if not edit_seq: # empty page?
            edit_seq.append(None)
        elif len(segment_text) >= max(vote.keys()) and edit_seq[-1] is not None:
            edit_seq.append(None) # this page longer than filter prediction, do more insert
        edit_seqs.append([seq.replace('- ', '') if seq else seq for seq in edit_seq])
    return edit_seqs


def sync_batch(input_ids: torch.Tensor, add_next:list[torch.LongTensor], match_idxes: list[int]) -> torch.Tensor:
    lst = [
            torch.cat([
                input_ids[i,:], #[input_ids[i,:]>2], 
                add_next[i][match_idxes[i]:]
            ]) for i in range(input_ids.size(0))
        ]
    ret = padded_stack(lst, side='left')
    return ret
