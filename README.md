# RNA Secondary Structure Prediction Using Machine Learning

IUP U-SOAR Summer 2026 Research Project

## Background

RNA secondary structure plays a critical role in determining RNA function, yet predicting how an RNA sequence folds remains a challenging problem in computational biology. This project investigates the use of machine learning models for RNA secondary structure prediction by representing structures as nucleotide contact maps. The research compares classical thermodynamic methods, such as ViennaRNA, against deep learning approaches including CNNs, BiLSTMs, and Transformer-based architectures. The ultimate goal is to improve structure prediction accuracy and contribute to a better understanding of the relationship between RNA sequence, structure, and function.

## Objectives

Develop and evaluate machine learning models for RNA secondary structure prediction using contact map prediction.

## Current Progress

- Dataset loader implemented
- Contact map generation from dot-bracket notation
- ViennaRNA baseline evaluation
- Evaluation metrics (Precision, Recall, F1)
- Visualization Pipeline

## Planned Models

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