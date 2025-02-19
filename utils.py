from transformers import AutoTokenizer, LayoutLMTokenizerFast
import torch
from PIL import Image
from io import BytesIO
from numpy import clip
from src.modules.utils import distance

import fitz
FLAGS = fitz.TEXTFLAGS_DICT | fitz.TEXT_DEHYPHENATE & ~fitz.TEXT_PRESERVE_IMAGES
import torch


def norm_text(text):
    text = text.replace('℃', 'C')
    text = text.replace('℉', 'F')
    text = text.replace('㎡', 'm')
    text = text.replace('ⅱ', 'i')
    text = text.replace('⒈', '1')
    text = text.replace('⒉', '2')
    text = text.replace('⒊', '3')
    text = text.replace('⒋', '4')
    text = text.replace('⒌', '5')
    text = text.replace('⒍', '6')
    text = text.replace('⒎', '7')
    text = text.replace('⒏', '8')
    text = text.replace('⒐', '9')
    text = text.replace('：', ':')
    text = text.replace('；', ';')
    text = text.replace('，', ',')
    text = text.replace('（', '(')
    text = text.replace('）', ')')
    text = text.replace('？', '?')

    return text


def extract_spans(pdf_page: fitz.Document):
    spans = []
    span_id = 0
    text_dict = pdf_page.get_text('dict', flags=FLAGS, sort=True)
    width = text_dict['width']
    height = text_dict['height']
    for block in text_dict['blocks']:
        if 'lines' in block:
            for line in block['lines']:
                for span in line['spans']:
                    span_bbox = span['bbox']
                    spans.append({
                        'box': span_bbox,
                        'text': norm_text(span['text']),
                        'id': span_id
                    })
                    span_id += 1
    return width, height, spans


class DataProcessor:
    def __init__(self,
                 tokenizer,
                 image_processor_nougat,
                 image_processor_layoutlmv3,
                 box_level='segment',
                 max_text_length=512,
                 norm_bbox_width=1000,
                 norm_bbox_height=1000) -> None:
        
        self.tokenizer = tokenizer
        self.image_processor_nougat = image_processor_nougat
        self.image_processor_layoutlmv3 = image_processor_layoutlmv3
        self.box_level = box_level
        self.dummy_bbox = [0, 0, 0, 0]
        self.max_text_length = max_text_length
        self.norm_bbox_width = norm_bbox_width
        self.norm_bbox_height = norm_bbox_height
    
    def process_file(self, pdf_filename):
        pdf_doc = fitz.open(pdf_filename)
        results = []
        for pdf_page in pdf_doc:
            result = {}
            input_ids = [self.tokenizer.cls_token_id]
            bboxes = [self.dummy_bbox]
            segment_idxs = [0]
            segment_text = ['']
            attention_mask = [1]
            width, height, spans = extract_spans(pdf_page)
            for segment in spans:
                segment_box = self.points_process(segment['box'],
                                              width,
                                              height,
                                              self.norm_bbox_width,
                                              self.norm_bbox_height)
                
                tokenized_results = self.tokenizer(
                    [segment['text']],
                    boxes=[segment['box']],
                    # truncation=False,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    return_attention_mask=False,
                    # is_split_into_words=False
                )

                seg_input_ids = tokenized_results['input_ids']
                segment_text.append(segment['text'])

                segment_idx = segment['id']
                for seg_input_id in seg_input_ids:
                    input_ids.append(seg_input_id)
                    attention_mask.append(1)
                    segment_idxs.append(segment_idx + 1)
                    bboxes.append(segment_box)
                
            assert len(input_ids) == len(bboxes)
            assert len(input_ids) == len(attention_mask)

            ## Truncation
            input_ids = input_ids[:self.max_text_length]
            attention_mask = attention_mask[:self.max_text_length]
            bboxes = bboxes[:self.max_text_length]
            segment_idxs = segment_idxs[:self.max_text_length]

            ## Padding
            pad_len = self.max_text_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            bboxes += [self.dummy_bbox] * pad_len

            result['input_ids'] = torch.tensor([input_ids])
            result['attention_mask'] = torch.tensor([attention_mask])
            result['segment_idxs'] = torch.tensor([segment_idxs + ([0] * pad_len)])
            result['position_2d'] = torch.tensor([bboxes])
            result['labels'] = None
            result['segment_text'] = [segment_text]

            ## image process
            pixmap = pdf_page.get_pixmap(dpi=150)
            stream = pixmap.pil_tobytes(format="png", optimize=True)
            img = Image.open(BytesIO(stream))

            result['images'] = torch.tensor(self.image_processor_layoutlmv3(img)['pixel_values'])
            result['pixel_values'] = torch.tensor(self.image_processor_nougat(img)['pixel_values'])
            results.append(result)
        return results
    
    def points_process(self, box, width, height, norm_width, norm_height):
        x0, y0, x1, y1 = box

        x0 = clip(0, int((x0 / width) * norm_width), norm_width)
        x1 = clip(0, int((x1 / width) * norm_width), norm_width)

        y0 = clip(0, int((y0 / height) * norm_height), norm_height)
        y1 = clip(0, int((y1 / height) * norm_height), norm_height)

        box = [x0, y0, x1, y1]

        return box


def compute_box_predictions(token_preds, token_in_bboxes, num_classes, token_labels=None):
    """
    Compute box-level predictions (and optionally labels) from token-level predictions.
    Also returns the number of tokens in each box.

    Args:
        token_preds (torch.Tensor): Tensor of shape [batch_size, max_len], containing the predicted class for each token.
        token_in_bboxes (torch.Tensor): Tensor of shape [batch_size, max_len], containing box IDs for each token (-100 for padding tokens).
        num_classes (int): The number of classes.
        token_labels (torch.Tensor, optional): Tensor of shape [batch_size, max_len], containing the true label for each token.

    Returns:
        torch.Tensor: Tensor of shape [batch_size, max_num_boxes], containing the predicted class for each box.
        torch.Tensor (optional): If token_labels is provided, returns a tensor of shape [batch_size, max_num_boxes], containing the true label for each box.
        torch.Tensor: Tensor of shape [batch_size, max_num_boxes], containing the number of tokens in each box.
    """
    batch_size, max_len = token_preds.size()
    mask = token_in_bboxes != -100  # Mask to identify valid tokens

    # Flatten batch and sequence dimensions while keeping valid tokens
    batch_indices = torch.arange(batch_size, device=token_preds.device).unsqueeze(1).expand(-1, max_len)
    batch_indices_flat = batch_indices[mask]  # Shape: [total_valid_tokens]
    token_preds_flat = token_preds[mask]
    box_ids_flat = token_in_bboxes[mask]

    # Create unique (batch_index, box_id) pairs
    indices = torch.stack([batch_indices_flat, box_ids_flat], dim=1)  # Shape: [total_valid_tokens, 2]
    unique_indices, inverse_indices = torch.unique(indices, dim=0, return_inverse=True)
    N_unique_boxes = unique_indices.size(0)

    # Compute counts of predicted classes per unique box
    C = num_classes
    offset_preds = inverse_indices * C + token_preds_flat
    counts_preds = torch.bincount(offset_preds, minlength=N_unique_boxes * C).view(N_unique_boxes, C)

    # Get the most frequent class per box for predictions
    box_preds = counts_preds.argmax(dim=1)

    # Compute the number of tokens per box
    token_counts = torch.bincount(inverse_indices, minlength=N_unique_boxes)

    # If token_labels is provided, compute box-level labels
    if token_labels is not None:
        token_labels_flat = token_labels[mask]
        offset_labels = inverse_indices * C + token_labels_flat
        counts_labels = torch.bincount(offset_labels, minlength=N_unique_boxes * C).view(N_unique_boxes, C)
        box_labels = counts_labels.argmax(dim=1)
    else:
        box_labels = None

    # Prepare the output tensors
    per_sample_box_ids = torch.full((batch_size, max_len), -1, dtype=torch.long, device=token_preds.device)
    per_sample_box_ids[mask] = box_ids_flat
    per_sample_max_box_ids = per_sample_box_ids.max(dim=1)[0]
    num_boxes_per_batch = per_sample_max_box_ids + 1
    max_num_boxes = num_boxes_per_batch.max()

    # Initialize the output tensors
    box_preds_tensor = torch.full((batch_size, max_num_boxes), 0, dtype=torch.long, device=token_preds.device)
    box_token_counts_tensor = torch.zeros((batch_size, max_num_boxes), dtype=torch.long, device=token_preds.device)
    if box_labels is not None:
        box_labels_tensor = torch.full((batch_size, max_num_boxes), -100, dtype=torch.long, device=token_preds.device)

    # Map box predictions, labels, and token counts to their respective positions in the output tensors
    unique_batch_indices = unique_indices[:, 0].long()
    unique_box_ids = unique_indices[:, 1].long()
    box_preds_tensor[unique_batch_indices, unique_box_ids] = box_preds
    box_token_counts_tensor[unique_batch_indices, unique_box_ids] = token_counts
    if box_labels is not None:
        box_labels_tensor[unique_batch_indices, unique_box_ids] = box_labels
        return box_preds_tensor, box_labels_tensor, box_token_counts_tensor
    else:
        return box_preds_tensor, box_token_counts_tensor