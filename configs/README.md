SynapseCLR Pretraining and Feature Extraction Configuration Files
=================================================================

This directory contains the configuration files for running SynapseCLR pre-training and feature extraction scripts.

Important Notes
===============

- Please edit `config.yaml` and update `dataset_path`, `checkpoint_path`, and `feature_output_path` according to your local paths.
- To continue a SynapseCLR pretraining run, set `reload` and `reload_optimizer_state` to `1` in `config.yaml`. 
- After successful SynapseCLR pretraining and before running SynapseCLR in the feature extraction mode, please edit `config.yaml`, set `reload` to `1`, and specify the model checkpoint index for feature extraction by modifying `reload_epoch_num`.
