# EditTrans
Edit-based Speedy Detailed Transformation of Academic Documents into Markup

Dataset is annotated by [LaTeX Rainbow](https://github.com/InsightsNet/texannotate).

## Repository Structure
```
root
├── data
│   ├── rainbow_bank           >>  sample data and arXiv ids to build the complete dataset
│   ├── data_maker.py          >>  make dataset from output of LaTeX Rainbow
│   └── data_spliter.py        >>  split dataset
├── filter_pth                 >>  fine-tuned LayoutLMv3 weight
├── nougat                     >>  implementation of metrics and Nougat model
├── edit_trains.py             >>  implementation of EditTrans
├── training_ernie.py          >>  fine-tuning code
├── requirements.txt
├── inference.py               >>  you may try EditTrans here
├── test.py                    >>  get scores for the baseline and EditTrans
└── utils.py
```

## Acknowledgments
This repository builds on top of the [Nougat](https://github.com/facebookresearch/nougat) and [Token-Path-Prediction](https://github.com/WinterShiver/Token-Path-Prediction) repository.