import json
from torchvision.datasets.folder import pil_loader
from numpy import clip
from ernie_layout_pytorch.networks import ErnieLayoutProcessor
import os


class ErnieProcessor:

    def __init__(self,
                 data_dir,
                 ernie_processor: ErnieLayoutProcessor,
                 max_length: int,
                 edit_label=False,
                 test=False
                 ):
        self.data_dir =data_dir
        self.ernie_processor = ernie_processor
        self.max_length = max_length
        self.edit_label=edit_label
        self.test = test

    def points_process(self, box, width, height, norm_width=1000, norm_height=1000):
        '''
        将4点坐标转为2点坐标
        这里直接使用左上、右下2点

        '''
        # 4点坐标不是完全的矩形而是和矩形很接近的四边形，所以使用左上0，右上1，右下2来近似计算框的h和w
        # h和w作为输入，用于模型CodeLayout，通过_calc_spatial_position_embeddings计算box的h, w对应的相对位置编码
        
        # 4点坐标
        x0, y0, x1, y1 = box

        x0 = clip(0, int((x0 / width) * norm_width), norm_width)
        x1 = clip(0, int((x1 / width) * norm_width), norm_width)

        y0 = clip(0, int((y0 / height) * norm_height), norm_height)
        y1 = clip(0, int((y1 / height) * norm_height), norm_height)

        box = [x0, y0, x1, y1]

        return box

    def process(self, sample, debug=False):
        texts = []
        layouts = []
        labels = []
        if isinstance(sample['json'], str):
            if sample['json'].startswith('/nfs/home/duan') and not os.path.isdir('/nfs/home/duan'): # fix path
                sample['json'] = self.data_dir + sample['json'][28:]
                sample['json'] = json.load(open(sample['json']))
            elif sample['json'].startswith('/pfss/mlde/users/cd58hofa/rainbow_bank/') and not os.path.isdir('/pfss/mlde/users/cd58hofa/rainbow_bank/'): # fix path
                sample['json'] = self.data_dir + sample['json'][39:]
                sample['json'] = json.load(open(sample['json']))
            elif sample['json'].startswith('{'): # fix path
                sample['json'] = json.loads(sample['json'])
            else:
                sample['json'] = json.load(open(sample['json']))
        d = sample['json']
        width = d['img']['width']
        height = d['img']['height']
        for seg in d['document']:
            texts.append(seg['text'])
            layouts.append(self.points_process(seg['box'], width, height))
            if seg['filter_label'] == 'D':
                labels.append(0)
            elif seg['filter_label'] == 'K' and self.edit_label:
                labels.append(2)
            else:
                labels.append(1)
        
        pil_image = pil_loader(os.path.join(self.data_dir, d['img']['path']))
        encoding = self.ernie_processor(
            pil_image, 
            texts, 
            boxes=layouts, 
            word_labels=labels, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        if not debug:
            for k in encoding.keys():
                encoding[k] = encoding[k][0]

        if self.test:
            encoding['texts'] = texts
        # encoding['texts'] = texts
        encoding['uid'] = d['uid']
        return encoding