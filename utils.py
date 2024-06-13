from transformers import AutoTokenizer, LayoutLMTokenizerFast
import torch
from PIL import Image
from io import BytesIO
from numpy import clip
from src.modules.utils import distance

import fitz
FLAGS = fitz.TEXTFLAGS_DICT | fitz.TEXT_DEHYPHENATE & ~fitz.TEXT_PRESERVE_IMAGES


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