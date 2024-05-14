
import argparse
from solver import Solver
from utils import str2bool
import numpy as np
import torch
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

def main():
    parser = parse_arguments()
    args, unparsed = parser.parse_known_args()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    net = Solver(args)
    net.train()


def parse_arguments():
    parser = argparse.ArgumentParser(description='training of vae')

    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
    parser.add_argument('--training_name', default='test', type=str,
                        help='a name for this training process, usually indicating training configuration')
    parser.add_argument('--reproducibility', default='True', type=str,
                        help='if True, then benchmarking of cudnn will be turned off and cudnn.deterministic will be set True')

    # Inits related to the model hyperparameters
    parser.add_argument('--z_dim', default=32, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=0.5, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--C_max', default=100, type=float, help='capacity parameter(C) of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
    parser.add_argument('--objective', default='geco', type=str, help='The default loss the the GECO loss from the GENESIS-V2 paper.')
    parser.add_argument('--model', default='BetaVAE_dstripes', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')

    # Inits related to the model and optimizer
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    parser.add_argument('--dataset', default='celeba_maskhq', type=str, help='dataset name')
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')

    parser.add_argument('--continue_w_ckpt', default='False', type=str,
                        help='if True, then the network will be initialized as saved in ckpt_dir/ckpt_name/last')
    parser.add_argument('--training_result_dir', default='results', type=str,
                        help='a directory to contain all the training output')

    # Inits related to model saving, loading and folder management
    parser.add_argument('--gamma', default=100, type=float,
                        help='gamma parameter for KL-term in understanding beta-VAE')
    parser.add_argument('--image_size', default=64, type=int, help='image size.')
    parser.add_argument('--num_workers', default=8, type=int, help='dataloader num_workers')
    parser.add_argument('--output_dir', default='outputs', type=str,
                        help='output directory for training visualization')
    parser.add_argument('--ckpt_name', default='last', type=str,
                        help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--loss_plot_epoch', default=1, type=int, help='plot loss every these many steps.')
    parser.add_argument('--save_epoch', default=20, type=int,
                        help='number of iterations after which a checkpoint is saved')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--dset_dir', default='datasets', type=str, help='dataset directory')
    
    # GECO loss configuration
    parser.add_argument('--g_goal', default = 0.0006, type = float, help = 'geco loss reconstruction goal (per pixel per channel).' )
    parser.add_argument('--g_lr', default = 1e-5, type = float, help = 'geco learning rate.')
    parser.add_argument('--g_alpha', default = 0.99, type = float, help = 'geco momentum for error.')
    parser.add_argument('--g_init', default = 1.0, type = float, help = 'geco inital Lagrange/beta factor.')
    parser.add_argument('--g_min', default = 1e-10, type = float, help  = 'geco min Lagrange/beta factor.')
    parser.add_argument('--g_speedup', default = 10., type = float, help = 'scale geco lr if delta positive.')


    return parser

if __name__ == "__main__":
    main()
