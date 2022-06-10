SynapseCLR Training and Feature Extraction
==========================================

Installation
------------

```
pip install -e pytorch_synapse --no-binary=connected-components-3d
```


Pretraining
-----------

To train on `N` nodes:
```
CUDA_VISIBLE_DEVICES=0 python ./synapse_simclr/synapse_simclr_pretrain.py --nr 0 --nodes N --config-yaml-path <config-path>
CUDA_VISIBLE_DEVICES=1 python ./synapse_simclr/synapse_simclr_pretrain.py --nr 1 --nodes N --config-yaml-path <config-path>
CUDA_VISIBLE_DEVICES=2 python ./synapse_simclr/synapse_simclr_pretrain.py --nr 2 --nodes N --config-yaml-path <config-path>
...
CUDA_VISIBLE_DEVICES=<N-1> python ./synapse_simclr/synapse_simclr_pretrain.py --nr <N-1> --nodes N --config-yaml-path <config-path>
```

Feature Extraction (after pretraining)
--------------------------------------

To extract features using `N` nodes:
```
CUDA_VISIBLE_DEVICES=0 python ./synapse_simclr/synapse_simclr_extract.py --nr 0 --nodes N --config-yaml-path <config-path>
CUDA_VISIBLE_DEVICES=1 python ./synapse_simclr/synapse_simclr_extract.py --nr 1 --nodes N --config-yaml-path <config-path>
CUDA_VISIBLE_DEVICES=2 python ./synapse_simclr/synapse_simclr_extract.py --nr 2 --nodes N --config-yaml-path <config-path>
...
CUDA_VISIBLE_DEVICES=<N-1> python ./synapse_simclr/synapse_simclr_extract.py --nr <N-1> --nodes N --config-yaml-path <config-path>
```

Notes
-----

- There are three YAML config files in the config folder; please reconfigure them before running pretraining and/or feature extraction (e.g. paths).

Misc
----

- To kill all python GPU processes:
```
nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9
```
