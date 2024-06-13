# 1. extract spans and bboxes
# 2. run the filter and predict tokens
# 3. vote for span's label and make Keep-Delete-Insert list
# 4. feed to Nougat!

from datasets import ClassLabel
from src.model.layoutlm_v3.configuration_layoutlmv3 import LayoutLMv3Config
from edit_trans import EditTrans
from nougat.utils.device import move_to_device
from transformers import AutoTokenizer, NougatProcessor, LayoutLMv3ImageProcessor
from utils import DataProcessor
from pathlib import Path


def inference(filename):
    labels = ClassLabel(names=['[DUMMY]', 'K', 'D', 'I'])
    filter_config = LayoutLMv3Config.from_pretrained('microsoft/layoutlmv3-base', output_hidden_states=True)
    filter_config.pretrained_model_path = 'microsoft/layoutlmv3-base'
    filter_config.adapter_pth_name = 'filter_pth'
    filter_config.num_labels = labels.num_classes
    filter_config.label2id = labels._str2int
    filter_config.id2label = labels._int2str
    filter_config.enable_position_1d = False

    pretrained_model = EditTrans(filter_config)
    pretrained_model = move_to_device(pretrained_model)

    pretrained_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='microsoft/layoutlmv3-base')
    processor_layoutlmv3 = LayoutLMv3ImageProcessor(apply_ocr=False)
    processor_nougat = NougatProcessor.from_pretrained("facebook/nougat-base")
    data_processor = DataProcessor(
        tokenizer,
        processor_nougat,
        processor_layoutlmv3
    )
    sample_pages = data_processor(filename)
    for i, sample in enumerate(sample_pages):
        output_name = Path(filename+'_'+str(i)+'.mmd')
        nougat_inputs = {'nougat_inputs': sample.pop('nougat_inputs')}
        outputs, steps = pretrained_model.inference(
            sample, nougat_inputs
        )
        outputs_text = pretrained_model.processor.batch_decode(
            outputs[0], skip_special_tokens=True
        )
        outputs_text = pretrained_model.processor.post_process_generation(outputs_text, fix_markdown=True)
        with open(output_name, 'w') as file:
            file.write(outputs_text)
    

if __name__ == "__main__":
    inference('test.pdf')
