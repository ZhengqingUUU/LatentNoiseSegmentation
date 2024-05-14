
import os
import sys
import shutil
from typing import Dict, List, Tuple, Optional
from colorama import Fore, Style

from utils import (cuda, versioning_folder, get_latest_version_folder, fprint,
                   plot_reconstruction)
from dataloaders import return_data
from models import *

from tqdm import tqdm
import torch
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from geco import *


class Solver:
    def __init__(self, args):
        super().__init__()
        # General inits
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0
        self.epoch_iter = 0
        self.training_name = args.training_name
        if args.reproducibility.lower() == 'true':
            try:
                torch.use_deterministic_algorithms(True)
            except:
                print("torch.use_deterministic_algorithms() not supported in current version of PyTorch.")
        self.train_loader, self.valid_loader = return_data(args)


        # Inits related to the model hyperparameters
        self.z_dim = args.z_dim
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model

        # Inits related to the model and optimizer
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        net, self.nc, self.is_VAE = self.load_model(args)
        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                betas=(self.beta1, self.beta2))

        # Geco loss config
        if self.objective.lower() == 'geco':
            # Goal is specified per pixel & channel so it doesn't need to
            # be changed for different resolutions etc.
            geco_goal = args.g_goal * self.nc * args.image_size**2
            # Scale step size to get similar update at different resolutions
            geco_lr = args.g_lr * (64**2 / args.image_size**2) 
            self.geco = GECO(geco_goal, geco_lr, args.g_alpha, args.g_init,
                        args.g_min, speedup = args.g_speedup, compute_on_cuda=self.use_cuda)
            self.beta = self.geco.beta # beta is set to 1 in this case.
        else: self.beta = args.beta 

        # Inits related to model saving, loading and folder management. Note, this should always be placed in
        # the last part of intialization.
        self.continue_with_checkpoints = True if args.continue_w_ckpt.lower() == 'true' else False
        self.training_outcome_savedir = self.get_savefolder(args)
        self.training_stdout_file=os.path.join(self.training_outcome_savedir, 'stdout.txt')
        self.ckpt_dir = os.path.join(self.training_outcome_savedir, args.ckpt_dir)
        self.output_dir = os.path.join(self.training_outcome_savedir, args.output_dir)
        self.save_output = args.save_output
        self.make_output_dirs(args)
        if args.ckpt_name is not None and self.continue_with_checkpoints: self.load_checkpoint(args.ckpt_name)
        elif args.ckpt_name is None and self.continue_with_checkpoints: fprint("no valid name for checkpoint to be loaded is provided", self.training_stdout_file)
        self.loss_plot_epoch= args.loss_plot_epoch
        self.save_epoch = args.save_epoch

        self.flag_dict = self.flags_to_dict(args)


    def train(self):
        # Initialize C for Burgess-style training, as in https://arxiv.org/pdf/1804.03599.pdf.
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))

        # Initialize progress bar
        out = False
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)

        # Initialize lists for loss plotting
        losses = {'train_loss': [], 'valid_loss': [],
                'train_recon': [], 'valid_recon': [],}
        if self.is_VAE: lantent_std_ls = []
        if self.objective.lower() == 'geco': beta_ls = []
        global_iter_intervals = []

        # Main loop
        fprint(f'Training started for {self.training_name}', self.training_stdout_file)
        while not out:
            self.epoch_iter += 1

            # Training pass
            self.net_mode(train=True)
            for x in self.train_loader:
                self.global_iter += 1
                pbar.update(1)

                # Feedforward pass and loss computation
                x = Variable(cuda(x, self.use_cuda))
                if self.is_VAE:
                    train_loss, train_recon_loss, _, train_std_arr = self.compute_loss(x, freeze_beta= False, )
                else:
                    train_loss, _ = self.compute_loss(x)
                    train_recon_loss = train_loss

                # Backward pass
                self.optim.zero_grad()
                train_loss.backward()
                epoch_train_loss = train_loss.item() 
                epoch_train_recon_loss = train_recon_loss.item() # we averaged over all the samples when calculating loss
                self.optim.step()

                if self.global_iter >= self.max_iter:
                    out = True
                    break

            # Validation pass
            self.net_mode(train=False)
            for x in self.valid_loader:
                x = Variable(cuda(x, self.use_cuda))
                if self.is_VAE:
                    valid_loss, valid_recon_loss, valid_recon,_ = self.compute_loss(x, freeze_beta = True) 
                else:
                    valid_loss, valid_recon = self.compute_loss(x)
                    valid_recon_loss = valid_loss
                epoch_valid_loss = valid_loss.item()
                epoch_valid_recon_loss = valid_recon_loss.item()

            # Plot print training information
            if (self.epoch_iter % self.loss_plot_epoch== 0) and not out:
                losses['train_loss'].append(epoch_train_loss)
                losses['valid_loss'].append(epoch_valid_loss)
                losses['train_recon'].append(epoch_train_recon_loss)
                losses['valid_recon'].append(epoch_valid_recon_loss)
                global_iter_intervals.append(self.global_iter)
                self.plot_losses(losses, global_iter_intervals,)
                if self.is_VAE:
                    lantent_std_ls.append(train_std_arr)
                    self.plot_latent_std(lantent_std_ls,global_iter_intervals)
                if self.objective.lower() == 'geco':
                    beta_ls.append(self.beta.cpu())
                    self.plot_geco_beta(beta_ls, global_iter_intervals)
                
                fprint(f"-epoch:{self.epoch_iter},iter:{self.global_iter},lr:{self.optim.param_groups[0]['lr']}-: "+
                    f"train loss: {losses['train_loss'][-1]:.4f}, "+
                    f"train recon. loss: {losses['train_recon'][-1]:.4f}, "+
                    f"valid loss: {losses['valid_loss'][-1]:.4f}, "+
                    f"valid recon. loss:{losses['valid_recon'][-1]:.4f}. ",
                    self.training_stdout_file)

            # Save most recent model, together with the reconstruction images 
            if self.epoch_iter % self.save_epoch == 0 or out:
                self.save_checkpoint('last')
                fprint(f'Saved checkpoint(iter: {self.global_iter})',self.training_stdout_file)

                # Save a reconstruction sample
                sampled_id = np.random.randint(x.shape[0])
                sampled_x, sampled_recon = x[sampled_id], valid_recon[sampled_id]
                self.plot_training_reconstruction(sampled_x, sampled_recon)

        pbar.write("[Training Finished]")
        pbar.close()

    def net_mode(self, train: bool):
        if train:
            self.net.train()
        else:
            self.net.eval()

    def compute_loss(self, x: Variable, freeze_beta = False, ):
        if self.is_VAE:
            x_recon, mu, logvar = self.net(x)
            recon_loss = reconstruction_loss(x, x_recon, )
            total_kld, dim_wise_kld, mean_kld,  std_arr = kl_divergence(mu, logvar, )
            loss = self.total_loss(recon_loss, total_kld, freeze_beta = freeze_beta)
            return loss, recon_loss, x_recon,  std_arr
        else: 
            x_recon, _  = self.net(x)
            recon_loss = reconstruction_loss(x,x_recon)
            return recon_loss, x_recon


    def total_loss(self, recon_loss: Tensor, total_kld: Tensor, freeze_beta = False) -> Tensor:
        if self.objective == 'H':
            loss = recon_loss + self.beta * total_kld
        elif self.objective == 'B':
            C = torch.clamp(self.C_max / self.C_stop_iter * self.global_iter, 0, self.C_max.data[0])
            loss = recon_loss + self.gamma * (total_kld - C).abs()
        elif self.objective.lower() == 'geco':
            self.beta = self.geco.beta # this beta is for future use, will be saved together with the model.
            loss = self.geco.loss(recon_loss, total_kld, compute_on_cuda=self.use_cuda,freeze_beta = freeze_beta)
        else:
            raise NotImplementedError
        return loss

    def plot_losses(self, losses: Dict[str, List[float]], global_iter_intervals: List[int],):
        plt.plot(global_iter_intervals, losses['train_loss'], label='Train loss')
        plt.plot(global_iter_intervals, losses['valid_loss'], label='Valid loss')
        plt.legend()
        plt.savefig(os.path.join(self.training_outcome_savedir, 'loss.png'))
        plt.clf()
        plt.plot(global_iter_intervals, losses['train_recon'], label='Train reconst.')
        plt.plot(global_iter_intervals, losses['valid_recon'], label='Valid reconst.')
        plt.legend()
        plt.savefig(os.path.join(self.training_outcome_savedir, 'recon_loss.png'))
        plt.clf()

    def plot_latent_std(self, latent_std_ls, global_iter_intervals):
        plt.plot(global_iter_intervals, latent_std_ls)
        plt.savefig(os.path.join(self.training_outcome_savedir, 'latent_std.png'))
        plt.clf()

    def plot_geco_beta(self, beta_ls: list, global_iter_intervals:list):
        plt.plot(global_iter_intervals, beta_ls)
        plt.savefig(os.path.join(self.training_outcome_savedir, 'beta_evolution.png'))
        plt.clf()
        
    def plot_training_reconstruction(self, input_image, reconstruction):
        """ Generate the right directory to place the reconstruction picture and then invoke plot_reconstruction()"""
        savedir = os.path.join(self.output_dir, 'reconstructions')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plot_reconstruction(input_image, reconstruction, savedir=savedir, image_name=str(self.global_iter)) 

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net': self.net.state_dict()}
        optim_states = {'optim': self.optim.state_dict()}
        geco_states = {'err_ema':self.geco.err_ema, 'beta': self.beta} if self.objective.lower()=='geco' else None

        states = {'iter': self.global_iter,
                  'model_states': model_states,
                  'optim_states': optim_states,
                  'geco_states': geco_states,
                  'flag_dict': self.flag_dict}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print(f'=> saved checkpoint {file_path} (iter {self.global_iter})', self.training_stdout_file)

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            if self.objective.lower()=='geco':
                self.geco.err_ema = checkpoint['geco_states']['err_ema']
                self.beta = checkpoint['geco_states']['beta']
            print(f'=> loaded checkpoint {file_path} (iter {self.global_iter})', self.training_stdout_file)
        else:
            print(f'=> no checkpoint found at {file_path}', self.training_stdout_file)

    def make_output_dirs(self, args):
        if not self.continue_with_checkpoints:
            # although versioning of folder is performed in get_savefolder funciton, we still 
            # do the following to err on the safe side
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
                fprint('output folder deleted', self.training_stdout_file)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


    def get_savefolder(self, args):
        """generate the savefolder for all the training results, versioning of folder included."""
        savefolder = os.path.join('.', args.training_result_dir, args.dataset, args.training_name)
        if not self.continue_with_checkpoints: # if not continue with checkpoints, then make sure to 
                                            # work inside another folder
            savefolder = versioning_folder(savefolder)
        else: # continue_with_checkpoints, then find the newest version ##TODO: test it! 
            savefolder = get_latest_version_folder(savefolder)
        if not os.path.exists(savefolder):
            os.makedirs(savefolder, exist_ok=True)
        return savefolder 

    def flags_to_dict(self, args=None):
        flag_dict = {}
        fprint(f"{Fore.YELLOW}HERE ARE THE INPUTTED FLAGS!!{Style.RESET_ALL}", self.training_stdout_file)
        for flag_name, val in vars(args).items():
            flag_dict[flag_name] = val
            fprint((flag_name, val), self.training_stdout_file)
        return flag_dict

    @staticmethod 
    def load_model(args):
        nc = 3
        try:
            net = eval(args.model)
        except:
            raise NotImplementedError(f'{args.model} is not a legitimate choice for model')
        is_VAE = False if not "VAE" in args.model else True
        return net, nc, is_VAE 


def reconstruction_loss(x, x_recon,):
    """return the mse loss
    """
    batch_size = x.size(0)
    x_recon = torch.sigmoid(x_recon)  # type: ignore
    recon_loss = F.mse_loss(x_recon, x, reduction="sum").div(batch_size)  # type: ignore
    return recon_loss


def kl_divergence(mu, logvar, ):
    batch_size = mu.size(0)
    assert batch_size != 0

    # reshape everything to 2 dimensional
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    # from page 5 of https://arxiv.org/pdf/1312.6114.pdf (original vae paper)
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    # record the std of all the latent nodes
    std_arr = np.mean(np.sqrt(np.exp(logvar.detach().cpu().numpy())),axis=0)

    return total_kld, dimension_wise_kld, mean_kld,  std_arr

