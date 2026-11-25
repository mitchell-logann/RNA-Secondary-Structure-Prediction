# RNA Secondary Structure Prediction Using Machine Learning

IUP U-SOAR Summer 2026 Research Project

## Background

RNA secondary structure plays a critical role in determining RNA function, yet predicting how an RNA sequence folds remains a challenging problem in computational biology. This project investigates the use of machine learning models for RNA secondary structure prediction by representing structures as nucleotide contact maps. The research compares classical thermodynamic methods, such as ViennaRNA, against deep learning approaches including CNNs, BiLSTMs, and Transformer-based architectures. The ultimate goal is to improve structure prediction accuracy and contribute to a better understanding of the relationship between RNA sequence, structure, and function.

## Objectives

Develop and evaluate machine learning models for RNA secondary structure prediction using contact map prediction.

## Models

- ViennaRNA
- CNN
- BiLSTM
- Transformer-based contact map predictor

## Dataset

bpRNA_1m dataset courtesy of Oregon State University

## Environment

Python 3.11
Pytorch
ViennaRNA
Biopython

## Methodology

Our training methodology was as followed:
- 60/20/20 Train/Val/Test split
- 768 nucleotide max sequence length
- 8 batch size
- 20 epochs
- Adam optimizer
- 0.001 learning rate
- Masked Binary Cross-Entropy loss

For evaluation, precision, recall, and F1 scores were used

Additionally, a graphics card configuration with 24 Gigabytes of VRAM was used for training.


## Results

A more detailed results section will be stored in the "Results" folder in the directory. Our findings are as follows:
- Transformer: 0.5047 Precision, 0.4894 Recall, 0.4928 F1
- CNN: 0.8006 Precision, 0.6347 Recall, 0.6723 F1
- **BiLSTM: 0.8472 Precision, 0.7356 Recall, 0.7636 F1**

We found that given our configuration, the BiLSTM model performed the strongest. This suggests that this architecture might be the strongest in this application, given the current configuration. 
