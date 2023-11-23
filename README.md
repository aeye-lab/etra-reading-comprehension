Inferring Native and Non-Native Human Reading Comprehension and Subjective Text Difficulty from Scanpaths in Reading
====================================================================================================================
[![paper](https://img.shields.io/static/v1?label=paper&message=download%20link&color=brightgreen)](https://dl.acm.org/doi/abs/10.1145/3517031.3529639)

This repo provides the code for reproducing the experiments in [Inferring Native and Non-Native Human Reading Comprehension and Subjective Text Difficulty from Scanpaths in Reading](https://dl.acm.org/doi/abs/10.1145/3517031.3529639).

![BEyeLSTM](https://user-images.githubusercontent.com/43832476/171489683-332d88ba-45f7-4f68-86dd-8288f52bd34c.png)
The figure above shows our architecture `BEyeLSTM`.
We investigate and show that we can generalize to unseen test persons for all tasks investigated.

## Reproduce the experiments

### Clone this repository
You can clone this repository by either using
```bash
git clone git@github.com:aeye-lab/etra-reading-comprehension
cd etra-reading-comprehension
```
or
```bash
git clone https://github.com/aeye-lab/etra-reading-comprehension
cd
```
depending on your preferences and settings.
Afterward, change into the directory by using `cd etra-reading-comprehension`.

### Download the data
You can download the publicly available data here
```bash
git clone git@github.com:ahnchive/SB-SAT
```
or
```bash
git clone https://github.com/ahnchive/SB-SAT
```

### Install packages
Install all required python packages via:
```bash
pip install -r requirements.txt
```
### Extract data
You can create the data splits using:
```bash
python3 utils/generate_text_sequence_splits.py
```

Then you can directly start using both BEyeLSTM and the baseline of [Ahn et al.](https://dl.acm.org/doi/10.1145/3379156.3391335) using `python3 nn/train_model.py` or `python3 ahn_baseline/evaluate_ahn_baseline.py` respectively. By changing the boolean arguments in `nn/train_model.py` you can recreate our ablation study or use different subnets only.

Note: Running the experiments, especially on CPU, will take some time.

## Contribute
If you find any issues, please open an issue in the issue tracker.

## Cite our work
If you use our code for your research, please consider citing our paper:

```bibtex
@inproceedings{10.1145/3517031.3529639,
author = {Reich, David Robert and Prasse, Paul and Tschirner, Chiara and Haller, Patrick and Goldhammer, Frank and J\"{a}ger, Lena A.},
title = {Inferring Native and Non-Native Human Reading Comprehension and Subjective Text Difficulty from Scanpaths in Reading},
year = {2022},
isbn = {9781450392525},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3517031.3529639},
booktitle = {2022 Symposium on Eye Tracking Research and Applications},
articleno = {23},
numpages = {8},
keywords = {deep learning, reading comprehension, eye tracking-while-reading},
location = {Seattle, WA, USA},
series = {ETRA '22}
}
```
