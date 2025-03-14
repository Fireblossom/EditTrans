import pandas as pd
import fitz
import re
import json
from operator import itemgetter
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
FLAGS = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES # fitz.TEXTFLAGS_DICT | fitz.TEXT_DEHYPHENATE & ~fitz.TEXT_PRESERVE_IMAGES
from helpers.multi_column import column_boxes
import pypandoc

@dataclass
class RainbowLatexToken:
    bbox: list[int]
    reading_order: int = -1
    label: Optional[str] = None
    block_id: int = -1
    section_id: int = -1
    text: str = ''
    line_no: int = 0
    tex: Optional[str] = None
    

def get_depth(toc_table: pd.DataFrame) -> dict:
    edges = []
    for i, row in toc_table.iterrows():
        if i > 0:
            edges.append((row['section_id'], row['nested_to']))

    node_depth = {node: -1 for node in set([n for edge in edges for n in edge])}
    node_depth[0] = 0

    # Perform a BFS traversal to update the depth of each node.
    queue = [0]
    while queue:
        current_node = queue.pop(0)
        for child in [edge[0] for edge in edges if edge[1] == current_node]:
            node_depth[child] = node_depth[current_node] + 1
            queue.append(child)

    return node_depth


def get_page_tokens(annotation_table, page_no) -> list[RainbowLatexToken]:
    tokens = []
    annotation_page = annotation_table[annotation_table['page_no']==page_no+1]
    for i, row in annotation_page.iterrows():
        bbox = [row['x0'], row['y0'], row['x1'], row['y1']]
        tokens.append(RainbowLatexToken(
            reading_order=row['reading_order'],
            label=row['label'],
            block_id=row['block_id'],
            section_id=row['section_id'],
            text=row['text'],
            line_no=row['line_no'],
            tex=row['tex'],
            bbox=bbox
        ))
    return tokens


def tokens_center_in_bbox(tokens: list[RainbowLatexToken], bbox):
    in_tokens = []
    x0, y0, x1, y1 = bbox
    for token in tokens:
        x0_, y0_, x1_, y1_ = token.bbox
        center_x = (x0_ + x1_)/2
        center_y = (y0_ + y1_)/2
        in_x = x0 < center_x < x1
        in_y = y0 < center_y < y1
        if in_x and in_y:
            in_tokens.append(token)
    return in_tokens


def block_center_in_rect(blocks, rect):
    x0, y0, x1, y1 = rect
    in_blocks = []
    for block in blocks:
        x0_, y0_, x1_, y1_ = block['bbox']
        center_x = (x0_ + x1_)/2
        center_y = (y0_ + y1_)/2
        in_x = x0 < center_x < x1
        in_y = y0 < center_y < y1
        if in_x and in_y:
            in_blocks.append(block)
    return in_blocks


def most_common(lst):
    return max(set(lst), key=lst.count)


def flags_decomposer(flags):
    """Make font flags human readable."""
    l = []
    if flags & 2 ** 0:
        l.append("superscript")
    if flags & 2 ** 1:
        l.append("italic")
    if flags & 2 ** 2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2 ** 3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2 ** 4:
        l.append("bold")
    return l


class DatasetMaker:
    def __init__(self, table_format='latex', table_position='end', figure_position='end'):
        self.table_format = table_format
        self.table_position = table_position
        self.figure_position = figure_position

    def process_file(self, filename: Path, output_path:Path):
        basepath = filename.parent
        annotation = pd.read_csv(basepath/(filename.stem+'_data.csv'), sep='\t', low_memory=False)
        annotation = annotation[annotation['label']!='Figure'] # no figures included
        toc = pd.read_csv(basepath/(filename.stem+'_toc.csv'), sep='\t')
        if len(annotation['section_id']) < 4: # bad annotation
            return
        self.node_depth = get_depth(toc)
        with fitz.open(filename) as pdf:
            self.caption_block_ids = set()
            jsonl_dataset = []
            for page_no, pdf_page_obj in enumerate(pdf.pages()):
                if page_no == 0:
                    self.first_page = True
                else:
                    self.first_page = False
                self.word_id = 0
                self.span_id = 0
                self.block_id = -1
                self.mmd_block_id = -1
                self.need_insert = False
                self.start_new_pdf_block_dataset = False
                self.start_new_pdf_block_mmd = False
                self.last_label_dataset = None
                self.last_2_label_dataset = None
                self.last_label_mmd = None
                self.last_2_label_mmd = None
                self.pdf_page_obj = pdf_page_obj
                self.reading_orders = {}
                self.add_end_reading_orders = {}
                self.delete_reading_orders = {}
                self.add_end_mmd = [] # '' resort after extraction
                dataset = []
                mmd = [] # ''
                
                page_pix = pdf_page_obj.get_pixmap(dpi=150)
                img_output_path = output_path/'images'
                img_output_path.mkdir(exist_ok=True)
                img_name = filename.stem+'_{}.png'.format(page_no)

                text_dict = pdf_page_obj.get_text('dict', flags=FLAGS, sort=True)
                
                text_rects = column_boxes(pdf_page_obj) 
                # sometimes doesn't work, so we resort with reading order from rainbowlatex
                # this step is time consuming, do we really need to keep it?

                height, width = text_dict['height'], text_dict['width']
                rainbow_tokens = get_page_tokens(annotation, page_no=page_no)
                labels_in_page = set([t.label for t in rainbow_tokens])
                if len(labels_in_page) <3 and 'Reference' in labels_in_page: # Reference only page
                    break

                for rect in text_rects:
                    in_rect_blocks = block_center_in_rect(text_dict['blocks'], rect) # sort by reading order in multi-column case
                    for block in in_rect_blocks:
                        self.start_new_pdf_block_dataset = True
                        self.start_new_pdf_block_mmd = True
                        block_bbox = block['bbox']
                        in_block_bbox_tokens = tokens_center_in_bbox(rainbow_tokens, block_bbox)
                        if 'Caption' in [t.label for t in in_block_bbox_tokens]:
                            self.caption_block_ids = self.caption_block_ids.union(set([t.block_id for t in in_block_bbox_tokens]))
                        if 'lines' in block:
                            for line in block['lines']:
                                for span in line['spans']:
                                    span_bbox = span['bbox']
                                    in_bbox_tokens = tokens_center_in_bbox(rainbow_tokens, span_bbox)
                                    if in_bbox_tokens:
                                        dataset += self.get_dataset(in_bbox_tokens, span)
                                        mmd += self.get_mmd(in_bbox_tokens, span)

                sorted_keys = sorted(self.reading_orders.items(), key=itemgetter(1))
                label_segment_order = [key for key, _ in sorted_keys]

                sorted_keys = sorted(self.add_end_reading_orders.items(), key=itemgetter(1))
                label_segment_order += [key for key, _ in sorted_keys]

                merged_reading_orders = {**self.reading_orders, **self.add_end_reading_orders, **self.delete_reading_orders}
                for k,v in merged_reading_orders.items():
                    if v == -1 and k-1 in merged_reading_orders:
                        merged_reading_orders[k] = merged_reading_orders[k-1] 
                        # fix -1 reading order, assume the same as previous
                        # TODO: should be the clostest one to the top?
                sorted_keys = sorted(merged_reading_orders.items(), key=itemgetter(1))
                sorted_dataset = [dataset[key] for key, _ in sorted_keys]

                if self.table_position == 'end' and self.figure_position == 'end':
                    sorted_mmd = [mmd[key] for key, _ in sorted_keys]
                    sorted_mmd += self.add_end_mmd
                elif self.table_position == 'start' and self.figure_position == 'end':
                    float_mmds = ''.join(self.add_end_mmd).split('\n\n')
                    add_start_mmd = []
                    add_end_mmd = []
                    for m in float_mmds:
                        if m.lower().lstrip('*').startswith('fig'): # figure caption
                            add_end_mmd.append('\n\n'+m)
                        else:
                            add_start_mmd.append('\n\n'+m)
                    sorted_mmd = add_start_mmd + [mmd[key] for key, _ in sorted_keys] + add_end_mmd
                else:
                    assert self.figure_position == 'middle' and self.table_position == 'middle', 'unk setting'
                    sorted_mmd = [mmd[key] for key, _ in sorted_keys]
                    
                sorted_mmd = [m[:-2] if m.endswith('- ') else m for m in sorted_mmd]
                mmd = ''.join(sorted_mmd)
                mmd = mmd.replace('\n\n\n\n', '\n\n')
                if mmd.startswith('\n\n'):
                    mmd = mmd[2:]
                # print(label_segment_order)
                json_d = {
                    'img': {'height': height, 'width': width, 'path': 'images/'+img_name},
                    'document': sorted_dataset,
                    'uid': img_name,
                    "label_segment_order": label_segment_order
                }
                if not mmd or len(mmd) < 50: # empty page
                    break

                mmd_output_path = output_path/'mmd'
                mmd_output_path.mkdir(exist_ok=True)
                mmd_name = filename.stem+'_{}.mmd'.format(page_no)
                with open(mmd_output_path/mmd_name, 'w') as mmd_file:
                    mmd_file.write(mmd)
                jsonl_dataset.append(json.dumps(json_d))
                page_pix.save(img_output_path/img_name)

            if jsonl_dataset:
                json_output_path = output_path/'jsonl'
                json_output_path.mkdir(exist_ok=True)
                with open(json_output_path/(filename.stem+'.jsonl'), 'w') as file_output:
                    file_output.write('\n'.join(jsonl_dataset))
    
    def get_dataset(self, in_bbox_tokens:list[RainbowLatexToken], span:dict):
        labels_set = set([t.label for t in in_bbox_tokens])
        if len(labels_set) == 1: # pure span
            if self.start_new_pdf_block_dataset:
                new_pdf_block = True
                self.start_new_pdf_block_dataset = False
            else:
                new_pdf_block = False
            if 'flags' in span:
                flags = flags_decomposer(span["flags"])
            else:
                flags = []
            if not len(set([t.block_id for t in in_bbox_tokens])) == 1:
                block_id = most_common([t.block_id for t in in_bbox_tokens])
            else:
                block_id = set([t.block_id for t in in_bbox_tokens]).pop()
            if self.block_id and block_id != self.block_id:
                # start new block
                new_block = True
                self.block_id = block_id
            else:
                new_block = False
            no_anno = False
            span_output = {}
            span_output['id'] = self.span_id
            self.span_id += 1
            span_output['text'] = span['text']
            span_output['box'] = list(span['bbox'])
            label = labels_set.pop()
            if label in ['Paragraph', 'Author', 'Reference', 'Abstract', 'Section', 'Title']:
                if self.need_insert:
                    span_output['filter_label'] = 'I'
                    self.need_insert = False
                elif new_block:
                    span_output['filter_label'] = 'I'
                else:
                    span_output['filter_label'] = 'K'

                if any(x in flags for x in ['italic', 'bold']):
                    span_output['filter_label'] = 'I'
                    self.need_insert = True

            elif label in ['Referece']:
                if self.need_insert:
                    span_output['filter_label'] = 'I'
                    self.need_insert = False
                elif new_block or new_pdf_block:
                    span_output['filter_label'] = 'I'
                else:
                    span_output['filter_label'] = 'K'
                
                if any(x in flags for x in ['italic', 'bold', 'superscript']):
                    span_output['filter_label'] = 'I'
                    self.need_insert = True

            elif label in ['List']:
                if self.need_insert:
                    span_output['filter_label'] = 'I'
                    self.need_insert = False
                elif new_block or '•' in span['text']:
                    span_output['filter_label'] = 'I'
                else:
                    span_output['filter_label'] = 'K'

                if any(x in flags for x in ['italic', 'bold', 'superscript']):
                    span_output['filter_label'] = 'I'
                    self.need_insert = True
                
            elif label in ['Footer']:
                if self.need_insert:
                    span_output['filter_label'] = 'I'
                    self.need_insert = False
                elif new_block or 'superscript' in flags: # footnote number
                    span_output['filter_label'] = 'I'
                else:
                    span_output['filter_label'] = 'K'
                
                if any(x in flags for x in ['italic', 'bold', 'superscript']):
                    span_output['filter_label'] = 'I'
                    self.need_insert = True

            elif label in ['Table', 'Equation', 'Caption']:
                span_output['filter_label'] = 'D'
                self.need_insert = True

            else: # no annotation spans
                span_output['filter_label'] = 'D'
                no_anno = True

            if isinstance(label, str):
                if label != self.last_label_dataset:
                    self.last_2_label_dataset = self.last_label_dataset
                    self.last_label_dataset = label
            if self.first_page and no_anno and 'Title' == self.last_2_label_dataset and 'Author' == self.last_label_dataset: # addresses
                span_output['filter_label'] = 'I'

            words = []
            for t in in_bbox_tokens:
                words.append({
                    'box': t.bbox,
                    'text': t.text,
                    'id': self.word_id
                })
                self.word_id += 1
            span_output['words'] = words
            if span_output['filter_label'] != 'D':
                #self.label_segment_order.append(span_output['id'])
                if not label in ['Table', 'Caption']:
                    self.reading_orders[span_output['id']] = in_bbox_tokens[round(len(in_bbox_tokens)/2)].reading_order
                else:
                    self.add_end_reading_orders[span_output['id']] = in_bbox_tokens[round(len(in_bbox_tokens)/2)].reading_order
            else:
                self.delete_reading_orders[span_output['id']] = in_bbox_tokens[round(len(in_bbox_tokens)/2)].reading_order
            # print(span_output['filter_label'], label)
            return [span_output]
        
        else:
            x = in_bbox_tokens
            indexes = [index for index, _ in enumerate(x) if x[index].label != x[index-1].label]
            indexes.append(len(x))
            splited = [x[indexes[i]:indexes[i+1]] for i, _ in enumerate(indexes) if i != len(indexes)-1]
            ret = []
            for s in splited:
                filtered_span = self.filter_span(s)
                ret += self.get_dataset(s, filtered_span)
            return ret

    def filter_span(self, splited_in_bbox_tokens):
        x0 = min([t.bbox[0] for t in splited_in_bbox_tokens])
        y0 = min([t.bbox[1] for t in splited_in_bbox_tokens])
        x1 = max([t.bbox[2] for t in splited_in_bbox_tokens])
        y1 = max([t.bbox[3] for t in splited_in_bbox_tokens])

        return{
            'bbox': [x0, y0, x1, y1],
            'text': self.pdf_page_obj.get_text("text",clip= fitz.Rect([x0, y0, x1, y1]), flags=FLAGS, sort=True)
        }

    def get_mmd(self, in_bbox_tokens:list[RainbowLatexToken], span:dict) -> str:
        labels_set = set([t.label for t in in_bbox_tokens])
        
        if len(labels_set) == 1: # pure span
            mmd = ''
            no_anno = False
            if self.start_new_pdf_block_mmd:
                new_pdf_block = True
                self.start_new_pdf_block_mmd = False
            else:
                new_pdf_block = False
            if 'flags' in span:
                flags = flags_decomposer(span["flags"])
            else:
                flags = []
            if not len(set([t.block_id for t in in_bbox_tokens])) == 1:
                block_id = most_common([t.block_id for t in in_bbox_tokens])
            else:
                block_id = set([t.block_id for t in in_bbox_tokens]).pop()
            if not len(set([t.section_id for t in in_bbox_tokens])) == 1:
                section_id = most_common([t.section_id for t in in_bbox_tokens])
            else:
                section_id = set([t.section_id for t in in_bbox_tokens]).pop()

            label = labels_set.pop()
            if block_id != self.mmd_block_id:
                # start new block
                new_block = True
                self.mmd_block_id = block_id
            elif {self.last_label_mmd, label} == {"Section", "Paragraph"}: # section and paragraph could have the same block id
                new_block = True
            else:
                new_block = False

            if label in ['Paragraph', 'Author']:
                if new_block and not mmd.endswith('\n\n'): 
                    mmd += '\n\n'
                if 'bold' in flags:
                    mmd += '**' + span['text'] + '** '
                elif 'italic' in flags:
                    mmd += '_' + span['text'] + '_ '
                else:
                    mmd += span['text'] + ' '
            elif label in ['Abstract']:
                if new_block:
                    mmd += '\n\n##### '
                    if not any(x in span['text'].lower() for x in ['abstract', 'key']):
                        mmd += 'Abstract\n'
                    else:
                        mmd += span['text'] + '\n'
                if 'bold' in flags:
                    mmd += '**' + span['text'] + '** '
                elif 'italic' in flags:
                    mmd += '_' + span['text'] + '_ '
                else:
                    mmd += span['text'] + ' '
            elif label in ['List']:
                if new_block and not mmd.endswith('\n\n'):
                    mmd += '\n\n'
                if 'bold' in flags:
                    mmd += '**' + span['text'] + '** '
                elif 'italic' in flags:
                    mmd += '_' + span['text'] + '_ '
                elif '•' in span['text']:
                    mmd += '\n* '
                elif re.search(r"^\d+(?:\s)?\.", span['text']):
                    mmd += '\n' + span['text']
                else:
                    mmd += span['text'] + ' '
            elif label in ['Reference']:
                if new_pdf_block:
                    mmd += '\n* '
                mmd += span['text'] + ' '
            elif label in ['Footer']:
                if 'superscript' in flags: # footnote number
                    mmd += '\n\nFootnote ' + span['text']
                elif 'bold' in flags:
                    mmd += '**' + span['text'] + '** '
                elif 'italic' in flags:
                    mmd += '_' + span['text'] + '_ '
                else:
                    mmd += span['text'] + ' '

            elif label in ['Section', 'Title']:
                if new_block and not mmd.endswith('\n\n'):
                    mmd += '\n\n'+'#'*self.node_depth[section_id]+' '

                mmd += span['text'] + ' '

            elif label in ['Equation']:
                if new_block and not mmd.endswith('\n\n'):
                    mmd += '\n\n'
                for t in in_bbox_tokens:
                    if isinstance(t.tex, str):
                        eq_mmd = t.tex.replace('\n', ' ')
                        eq_mmd = re.sub(r"\]\s\(\d+\.\d+\)", "", eq_mmd)
                        if t.block_id in self.caption_block_ids:
                            self.add_end_mmd.append(eq_mmd)
                        else:
                            mmd += eq_mmd
            elif label in ['Table']:
                if new_block and self.add_end_mmd and not self.add_end_mmd[-1].endswith('\n\n'):
                    self.add_end_mmd.append('\n\n')
                elif new_block and self.table_position == 'middle':
                    self.add_end_mmd.append('\n\n')

                for t in in_bbox_tokens:
                    if isinstance(t.tex, str):
                        if self.table_format == 'latex':
                            self.add_end_mmd.append(t.tex)
                        elif self.table_format == 'md':
                            self.add_end_mmd.append(pypandoc.convert_text(t.tex, 'md', format='latex'))
                        elif self.table_format == 'html':
                            self.add_end_mmd.append(pypandoc.convert_text(t.tex, 'html', format='latex'))
                        else:
                            raise ValueError('Table format not supported.')
                        break
                            
            elif label in ['Caption']:
                # add to end
                if new_block and self.add_end_mmd and not self.add_end_mmd[-1].endswith('\n\n'): 
                    self.add_end_mmd.append('\n\n')
                elif new_block and self.table_position == 'middle':
                    self.add_end_mmd.append('\n\n')

                if 'bold' in flags:
                    self.add_end_mmd.append('**' + span['text'] + '** ')
                elif 'italic' in flags:
                    self.add_end_mmd.append('_' + span['text'] + '_ ')
                else:
                    self.add_end_mmd.append(span['text'] + ' ')
            else: # no annotation span
                no_anno = True

            if isinstance(label, str):                    
                if label != self.last_label_mmd:
                    self.last_2_label_mmd = self.last_label_mmd
                    self.last_label_mmd = label
            if self.first_page and no_anno and 'Title' == self.last_2_label_mmd and 'Author' == self.last_label_mmd: # addresses
                if new_block and not mmd.endswith('\n\n'): 
                    mmd += '\n\n'
                if not 'Abstract' in span['text']:
                    if 'superscript' in flags:
                        mmd += '\n' + span['text'] + ' '
                    elif 'bold' in flags:
                        mmd += '**' + span['text'] + '** '
                    elif 'italic' in flags:
                        mmd += '_' + span['text'] + '_ '
                    else:
                        mmd += span['text'] + ' '

            if self.table_position == 'middle' and self.add_end_mmd:
                mmd += ''.join(self.add_end_mmd)
                self.add_end_mmd = []
            return [mmd]
        
        else:
            x = in_bbox_tokens
            indexes = [index for index, _ in enumerate(x) if x[index].label != x[index-1].label]
            indexes.append(len(x))
            splited = [x[indexes[i]:indexes[i+1]] for i, _ in enumerate(indexes) if i != len(indexes)-1]
            ret = []
            for s in splited:
                filtered_span = self.filter_span(s)
                ret += self.get_mmd(s, filtered_span)
            return ret


def process_file(file, output, table_format='latex', table_position='end', figure_position='end'):
    processor = DatasetMaker(table_format, table_position, figure_position)
    processor.process_file(file, output)


def main(input_path, output_path, table_fromat, table_position, figure_position):
    from pebble import ProcessPool
    from multiprocessing import set_start_method
    from concurrent.futures import TimeoutError, as_completed
    from collections import OrderedDict
    import logging

    TIMEOUT_SECONDS = 1200
    
    output_path.mkdir(exist_ok=True)
    
    logging.basicConfig(filename='error.log', encoding='utf-8', level=logging.ERROR)
    logger = logging.getLogger(name=None)
    logger.setLevel(logging.ERROR)

    args = []
    for file in input_path.glob('*.pdf'):
        #file = Path('outputs/1412.8507.pdf')
        args.append([file, output_path, table_fromat, table_position, figure_position])

    debug = False
    if debug:
        for arg in args:
            process_file(*arg)

    with ProcessPool(max_workers=96) as pool:
        try:
            tasks = OrderedDict()
            for arg in args:
                future = pool.schedule(process_file, args=arg, timeout=TIMEOUT_SECONDS)
                tasks[future] = arg
            
            finished = 0
            num_tasks = len(tasks)
            for future in as_completed(tasks):
                finished += 1
                try:
                    r = future.result()  # blocks until results are ready
                except TimeoutError as error:
                    logger.error(tasks[future][0], 'Timeout.') 
                except Exception as e:
                    logger.error(tasks[future][0], e)
                print("Finished task:{0}/{1}".format(finished, num_tasks))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            pool.close()

if __name__ == "__main__":
    #process_file(Path('outputs_latex_rainbow/2307.05191.pdf'), Path('.')) # debug

    input_path = Path('outputs_latex_rainbow/')
    output_path= Path('rainbow_bank_nougat')
    main(input_path, output_path, 'latex', 'end', 'end')

    output_path = Path('rainbow_bank_kosmos')
    main(input_path, output_path, 'html', 'start', 'end')

    output_path = Path('rainbow_bank_olmocr')
    main(input_path, output_path, 'md', 'middle', 'middle')