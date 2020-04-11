# Music Representation Learning from Tags and Unstructured Text

## Authors

* Alexander Schindler
* Peter Knees
* Navid
* Markus Schedl
* Sergiu Gordea

## Publications

* Alexander Schindler, Sergiu Gordea and Peter Knees. Unsupervised Cross-Modal Audio Representation Learning from Unstructured Multilingual Text. In Proceedings of the 35th ACM/SIGAPP Symposium On Applied Computing (SAC2020), March 30-April 3, 2020, Brno, Czech Republic.

* Alexander Schindler and Peter Knees. Multi-Task Music Representation Learning from Multi-Label Embeddings. In Proceedings of the International Conference on Content-Based Multimedia Indexing (CBMI2019). Dublin, Ireland, 4-6 Sept 2019.


# Pre-trained models

## Music Representation Backbone Models

One Model for each experiment

## Music Simlarity Retrieval

## Music Classification




# Reproducing the Experiments

## Prerequisites

* MSD audio samples
* MSD AMG Tagsets
** Tagsets can be downloaded from https://github.com/tuwien-musicir/msd/tree/master/ground_truth_assignments/AMG_Multilabel_tagsets
** for the experiments the hdf5 archive "msd_amglabels_all.h5" is used
* MSD AMG Album Reviews

## Align your Data

* Execute Notebook: 1 - Prepare Experiment Metadata.ipynb
** Supply configuration values in first cell
** execute all remaining cells

This notebook 
* aligns your MSD audio samples with the AMG Tagsets and the AMG album reviews.
* creates parition files (train, validation) with trackids and filenames for the feature extraction step

Published experimental results:
* In directory ..paper.. we provide our mappings for comparison

## Extract Mel-Spectrograms

* extract Mel-Spectrograms for the audio models

TODO: example script 

## 