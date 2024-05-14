#!/bin/sh

cd ..


python ./train.py --dataset dClosure --model AE_model --training_name dClosure_AE_1\
 --seed 1 --lr 5e-5 --beta1 0.9 --beta2 0.999 --batch_size 64 --z_dim 15 --max_iter 1.1e6\
 --save_output True --continue_w_ckpt False --num_workers 8 --reproducibility True --save_epoch 400\
 --objective geco  --g_goal 0.0006
