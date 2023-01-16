# SynapseCLR
SynapseCLR is a contrastive learning framework for navigating 3D electron microscopy data. A graphical overview of SynapseCLR pipeline and downstream applications is shown below:

![Alt text](./docs/source/_static/synapseclr_graphical_abstract_1200x1200.png "SynapseCLR Overview")

# Navigating this Repository
SynapseCLR repository is organized as follows:
```
<repo_root>/
├─ pytorch_synapse/       # SynapseCLR Python packages
├─ configs/               # Sample configurations for pretraining SynapseCLR models
├─ data/                  # Processed 3D EM image chunks
├─ ext/                   # External resources (e.g. other pretrained models)
├─ output/                # SynapseCLR outputs (model weights, interactive analysis results)
├─ scripts/               # Helper scripts
├─ tables/                # Input and generated DataFrames
└─ notebooks/             # Notebooks for data pre-processing, interactive analysis, and reproducing paper figures
```

If you wish to _explore_ the results, a good starting point is browsing `notebooks` in GitHub. If you wish to _run_ the notebooks, you need to install `pytorch_synapse` and additionally download the contents of `data` and `output`. Instructions for downloading the data and pretrained models will be provided shortly. Finally, if you wish to _pretrain_ SynapseCLR on your own 3D EM image chunks (not necessarily synapses, mind you; mitochondria anyone?), please follow the instructions given in `pytorch_synapse`. You will need to preprocess your data as described in `notebooks/01_data_preprocessing` and modify the code sightly. Feel free to contact us!

# Data Download
You can download SynapseCLR preprocessed data, pretrained models, and analysis results from the public Google bucket `gs://broad-dsp-synapseclr-data`. Please visit [here](https://cloud.google.com/storage/docs/uploads-downloads) to learn more about downloading data from Google buckets. The bucket includes the content of the following folders:
```
<repo_root>/
├─ data/                  # Processed 3D EM image chunks
├─ ext/                   # External resources (e.g. other pretrained models)
├─ output/                # SynapseCLR outputs (model weights, interactive analysis results)
└─ tables/                # Input and generated DataFrames
```

# Preprint and Citation
The bioRxiv preprint for SynapseCLR can be found [here](https://www.biorxiv.org/content/early/2022/06/09/2022.06.07.495207). The BibTeX citation is as follows:
```
@article {Wilson2022.06.07.495207,
	author = {Wilson, Alyssa M and Babadi, Mehrtash},
	title = {Uncovering features of synapses in primary visual cortex through contrastive representation learning},
	elocation-id = {2022.06.07.495207},
	year = {2022},
	doi = {10.1101/2022.06.07.495207},
	URL = {https://www.biorxiv.org/content/early/2022/06/09/2022.06.07.495207},
	eprint = {https://www.biorxiv.org/content/early/2022/06/09/2022.06.07.495207.full.pdf},
	journal = {bioRxiv}
}
```

# Authors
- Alyssa M. Wilson <alyssa.wilson@mssm.edu> (Icahn School of Medicine at Mount Sinai, New York, NY)
- Mehrtash Babadi <mehrtash@broadinstitute.org> (Data Sciences Platform, Broad Institute, Cambridge, MA)
