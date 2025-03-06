# EditTrans
Edit-based Speedy Detailed Transformation of Academic Documents into Markup

## Repository Structure
```
root
├── data
│   ├── rainbow_bank           >>  sample data and arXiv ids to build the complete dataset
│   ├── data_maker.py          >>  make dataset from output of LaTeX Rainbow
│   └── data_spliter.py        >>  split dataset
├── filter_pth                 >>  fine-tuned ERNIE Layout weight
├── training_ernie.py          >>  fine-tuning filter
├── nougat                     >>  implementation of metrics from Nougat model
├── edit_trains                >>  implementation of EditTrans
├── results                    >>  latency, step and scores in paper
└── test_*.py                  >>  get scores for the baseline and EditTrans
```

## Acknowledgments
Dataset is annotated by [LaTeX Rainbow](https://github.com/InsightsNet/texannotate).

This repository builds on top of the [Nougat](https://github.com/facebookresearch/nougat) and [Token-Path-Prediction](https://github.com/WinterShiver/Token-Path-Prediction) repository.