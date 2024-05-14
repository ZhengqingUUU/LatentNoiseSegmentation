#!/bin/sh

cd ..

python ./experiments.py --dataset dClosure --training_name dClosure_AE_1\
    --model AE_model --z_dim 15 --test_num 100 --n_clusters -1 --noise_sample_std 0.001\
    --perturb_num 40 --require_UMAP --UMAP_num 20 --ckpt_name last