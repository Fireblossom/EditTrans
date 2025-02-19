# %%
from typing import List
from transformers.generation.stopping_criteria import StopStringCriteria
from transformers import AutoTokenizer
from torch.nn import functional as F
import torch
# %%
tokenizer = AutoTokenizer.from_pretrained('facebook/nougat-base')

# %%
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

stop = PerRowStopStringCriteria(tokenizer, stop_strings=['abc', 'def'])

# %%
tokenizer.padding_side='left'
input_ids = tokenizer(['this is abc', ' this is a def', 'this not def'], return_tensors='pt', padding=True).input_ids

# %%
input_ids

# %%
print(stop(input_ids[:, :-1].cuda(), scores=None))

# %% [markdown]
# 

# %%
stop.max_valid_end_lens

# %%
token_list, token_indices = stop.clean_tokenizer_vocab(tokenizer)

# %%
token_valid_positions, token_end_overlaps = stop._stop_string_get_matching_positions(token_list, token_indices, ['abc'])


