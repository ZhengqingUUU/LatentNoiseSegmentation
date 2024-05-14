# VAE_segmentation_GoodGestalt

This is the code base for the paper [Latent Noise Segmentation: How Neural Noise Leads to  the Emergence of Segmentation and Grouping](https://arxiv.org/pdf/2309.16515).

## Step 0: Installing Required Libraries

```
pip install requirements.txt
```

## Step 1: Generating the Datasets

```bash
cd datasets
python create_datasets.py
```

The above generates all the 6 Good Gestalt (GG) datasets required. 

After the GG datasets are generated, one can browse the example training/validation/testing images in `./datasets`. Each dataset is saved in one folder. The names of the folders and the datasets match correspondingly:

```bash
dClosure === Closure
dContinuity === Continuity
dIlluOcclusion === Illusionary Occlusion
dGradOcclusion === Gradient Occlusion
dKanizsa === Kanizsa Squares
dProximity === Proximity
```

![](C:\Users\Zhengqing\AppData\Roaming\marktext\images\2024-05-14-16-57-18-image.png)

Each dataset above will have train/validation/test datasets, saved in `train.npz`, `valid.npz`, `test.npz` (`test_1.npz`), respectively. `test_1.npz` are testing images sampled from the same distribution as the training and validation images. These testing images are of less interest for our project and are not used for now. `test.npz` are testing images specially designed for inducing special visual effects (illusions). 

If you desire to investigate the models performance on the CelebA dataset, please download the [CelebAMask-HQ folder](https://github.com/switchablenorms/CelebAMask-HQ) and put it in the `datasets` folder, then add `--include_celeba True` when running `create_datasets.py`.  The adapted CelebA dataset that suits our model will be stored in a folder  called `dCeleba`, with a downsampled resolution and customized masks. 

If you prefer images with higher resolution (256 $\times$ 256, instead of 64 $\times$ 64), please add `--high_res True`.

## Step 2: Training the Models

```bash
python ./train.py --dataset [DATASET] --model [MODEL] --training_name [TRAINING_NAME]\
 --seed [SEED] --lr 5e-5 --beta1 0.9 --beta2 0.999 --batch_size 64 --z_dim 15 --max_iter 1.1e6\
 --save_output True --continue_w_ckpt False --num_workers 8 --reproducibility True --save_epoch 400\
 --objective geco  --g_goal 0.0006
```

The training process can be started by invoking the above clause. One can refer to `train.py` to understand the configurable parameters. To enable the training for different datasets and models, one need to specify the following fields as shown in the snippet above to start training:

1. `[DATASET]` is the name of the dataset you wish to train the model on. This field, by default, can be `dClosure`, `dContinuity`, `dContour`, `dGradOcclusion`, `dKanizsa`, `dProximity`, or `dCeleba`. 

2. `[MODEL]` specifies whether to learn the data with a autoencoder(AE) model or a beta-variational-autoencoder (beta-VAE) model. To invoke the former, this field should be `AE_model`, and to invoke the latter, this field should be `BetaVAE_model`.

3. `[TRAINING_NAME]` registers an indentifier for this training process. The data generated during the training process will be stored in `./results/[DATASET]/[TRAINING_NAME]`. This field can be an arbitrary string.

4. `[SEED]` is the random seed to ensure reproducibility of the training process.

<span style="color:green"> An example of the training script is provided in </span>`./scripts/train.sh`.

One reminder: the hyperparameters for training on CelebA dataset are different from the rest of the datasets, please refer to [our paper](https://arxiv.org/pdf/2309.16515).

## Step 3: Performing Segmentation

```bash
python ./experiments.py --dataset [DATASET] --training_name [TRAINING_NAME]\
    --model [MODEL] --z_dim 15 --test_num [TEST_NUM] --n_clusters [N_CLUSTERS] --noise_sample_std [STD]\
    --perturb_num [NUM] --require_UMAP --UMAP_num [UMAP_NUM] --ckpt_name last
```

The above script will start the tesing process. The configurable variables include:

1. `[DATASET]`, `[TRAINING_NAME]`, `[MODEL]` are explained in **Step 2** above. 

2. `[TEST_NUM]` is the number of testing samples the testing process will make use of. This number should be ≤ 100, if you are using the default setting of dataset generation in **Step 1**, since we generate 100 training samples for each dataset.

3. `[N_CLUSTERS]` is a how many clusters you wish the segmentation pipeline to group the pixels into. Setting this field as `-1` is equivalent to choosing the ground truth cluster number for each testing sample, which is implicitly stored in the masks.

4. `[STD]` is the standard deviation of the zero-centered Gaussian noise with which we perturb the latnt values to generate the segmentation masks.

5. `[NUM]` is how many rounds of perturbation will be carried out to acquire the semgentation masks.

6. `[UMAP_NUM]` specifies how many testing samples will be put through the UMAP visualization process to see how the pixels are represented in the high-dimensional space where the clustering is performed. UMAP can be quite slow, so one can accelerate the testing process by setting `[UMAP_NUM]` as a small number such that only the first a few testing samples will undergo the UMAP process.

<span style="color:green"> An example of the testing script is provided in </span>`./scripts/test.sh`.

## License

This project is licensed under the GPL v3.0 License - see the [LICENSE](LICENSE.txt) file for details.

## Citation

If you make use of this code in your research, we would appreciate if you considered citing the paper:

```
@misc{lonnqvist2024latent,
      title={Latent Noise Segmentation: How Neural Noise Leads to the Emergence of Segmentation and Grouping}, 
      author={Ben Lonnqvist and Zhengqing Wu and Michael H. Herzog},
      year={2024},
      eprint={2309.16515},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Third-party Code

This repository includes a modified version of `geco.py`, which was originally from [GENESIS](https://github.com/applied-ai-lab/genesis?tab=readme-ov-file#license), licensed with GPL v3.0.
