
import os
import argparse
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import torch
import umap.umap_ as umap
import torch.nn as nn
from torch import Tensor
from segmentation import NoisySegmentation
from sklearn.metrics.cluster import adjusted_rand_score
from models import *
import pandas as pd
from utils import cuda
from tqdm import tqdm
from dataloaders import return_data
from umap_plotting import *


class SegmentationTester:
    """General class for segmenting, in charge of feeding data into a NoisySegmentation object,
    collecting data generated from the segmentation process."""
    def __init__(self, args, model: nn.Module, save_directory: str, clustering_args: Dict[str, str], cuda: bool):
        self.device = 'cuda' if cuda else 'cpu'
        self.segmenter = NoisySegmentation(model=model,
                                           perturb_num=args.perturb_num,
                                           clustering_args=clustering_args,
                                           device=self.device,
                                           image_size=args.image_size,
                                           noise_sample_distribution=args.noise_sample_distribution,
                                           noise_sample_std=args.noise_sample_std)
        
        self.dataset = args.dataset
        self.dataloader, self.mask, self.detailed_mask = return_data(args, return_testset=True)  # return_data returns a list,
                                                                     #  with only the test loader
        self.dataloader = self.dataloader[0]
        self.save_directory = save_directory
        self.require_UMAP, self.UMAP_html_plot ,self.UMAP_n_neighbors, self.UMAP_num = args.require_UMAP,args.UMAP_html_plot, args.UMAP_n_neighbors, args.UMAP_num
        self.require_recon_plot = args.require_recon_plot

    def run_segmentation_tests(self, num_segmentations_per_model: int = 10):
        performance_array = np.zeros((0,3))
        prediction_list = []
        reconstruction_list = []
        umap_df_list = []
        for i, (x, m) in tqdm(enumerate(zip(self.dataloader,self.mask)),total=num_segmentations_per_model):
            m = m.squeeze() # deletable, just in case.
            right_cluster_num = np.unique(m).shape[0]
            reconstruction, prediction, recon_perturbation = self.segmenter(x.to(self.device), right_cluster_num)
            self.save_output(x,reconstruction,prediction,savedir=self.save_directory, image_id=i)
            prediction_list.append(np.expand_dims(prediction, axis = 0))
            reconstruction_list.append(np.expand_dims(reconstruction.cpu().detach().numpy(), axis = 0))
            
            ari, ari_fg = self.compute_ari(prediction = prediction, mask = m)

            # compile ari information into an array, which will later form a dataframe. 
            # note: here, we also compute foreground_ari (ari_fg), which is not used in the end.
            one_row_info = np.asarray([[i,ari,ari_fg]],dtype=float)
            performance_array = np.concatenate((performance_array,one_row_info),axis=0)

            # perform UMAP
            if self.require_UMAP and i < self.UMAP_num:
                umap_df = self.perform_umap(recon_perturbation = recon_perturbation, input = x, mask = m, index =i)
                umap_df_list.append(umap_df)

            # save the dataframe
            if i >= num_segmentations_per_model-1:
                performance_df = pd.DataFrame(performance_array, columns=['index', 'ARI', 'ARI_FG'])
                performance_df.to_pickle(os.path.join(self.save_directory,'performance_dataframe.pkl'))
                break
            if i == 0: print("clustering process operated correctly once.")

        prediction_array = np.concatenate(prediction_list, axis = 0).squeeze()
        np.savez_compressed(os.path.join(self.save_directory, 'prediction.npz'), prediction = prediction_array)
        reconstruction_array = np.concatenate(reconstruction_list, axis = 0).squeeze()
        np.savez_compressed(os.path.join(self.save_directory, 'reconstruction.npz'), reconstruction= reconstruction_array)
        
        umap_df_all = pd.concat(umap_df_list, axis=0) 
        umap_df_all.to_pickle(os.path.join(self.save_directory, 'umap_df.pkl'))

    def compute_ari(self, prediction, mask):
        """compute ARI and foreground-ARI"""

        # calculate ARI, ARI-FG
        ari = adjusted_rand_score(prediction.reshape(-1),mask.reshape(-1))
        
        # note, when m is single class, ari does not make much sense at all!
        fg_mask = mask!=0
        m_fg = mask[fg_mask]
        prediction_fg = prediction[fg_mask]
        ari_fg = adjusted_rand_score(prediction_fg, m_fg)

        return ari, ari_fg

    def perform_umap(self, recon_perturbation, input, mask, index, ):

        pixel_num = recon_perturbation.shape[-1]*recon_perturbation.shape[-2]
        to_be_umapped_data = recon_perturbation.reshape(-1,pixel_num).T # all perturbation of one pixel in one row.

        umapped_data = UMAP_to_2D(to_be_umapped_data) # (pixel_num,2)
        umap_df = pd.DataFrame(umapped_data, columns= ['UMAP_dim1','UMAP_dim2'])

        x_0 = input[0,0].reshape(pixel_num,1)
        umap_df.insert(0,'orig_x_1',x_0)

        # generate a column 'label' to indicate the label of all the pixels
        if self.detailed_mask.ndim != 0 : # some datasets do not have detailed masks
            detailed_mask = self.detailed_mask[index].reshape(pixel_num,-1) # should be one column
        else:
            detailed_mask = mask.reshape(pixel_num,-1)
        umap_df.insert(0,'label_num', detailed_mask)
        def to_str(num): # Just to technical convenience. More stable in this way.
            return str(int(num))
        umap_df['label'] = umap_df.apply(lambda row: to_str(row.label_num), axis = 1)

        # plot UMAP result No.1, a jpg image
        UMAP_plot(umap_df=umap_df, savedir = self.save_directory, image_id = index, dataset = self.dataset)

        # generate a column including the coordinates of all the pixels.   
        xy_list = [str(j)+'_'+str(i) for j in range(recon_perturbation.shape[-2])\
                    for i in range(recon_perturbation.shape[-1]) ]
        umap_df.insert(0, 'xy', np.array(xy_list, dtype=object).reshape(-1,1))

        if self.UMAP_html_plot:
            # plot UMAP result No.2, an html image 
            UMAP_interactive_plot(umap_df = umap_df, savedir = self.save_directory, 
                                            image_id = index, dataset = self.dataset)
        umap_df.insert(0,'image_id', index)
        return umap_df

    def save_output(self, original, reconstruction, output: Tensor, savedir: str, image_id: int):

        # plot the original picture
        original = original.detach().cpu().numpy()[0] 
        original = np.einsum('kij->ijk',original).squeeze()
        plt.imshow(original)
        plt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, str(image_id) + '_orig.jpg'))
        plt.clf()

        if self.require_recon_plot: 
            # plot the reconstructed picture
            reconstruction = np.einsum('kij->ijk',reconstruction.cpu().detach().squeeze())
            plt.imshow(reconstruction)
            plt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            os.makedirs(savedir, exist_ok=True)
            plt.savefig(os.path.join(savedir, str(image_id) + '_recon.jpg'))
            plt.clf()

        # plot the segmentation mask
        plt.imshow(output)
        plt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(os.path.join(savedir, str(image_id) + '_seg.jpg'))
        plt.clf()


def UMAP_to_2D(data_for_umap ,n_neighbors=400):
    reducer = umap.UMAP(n_neighbors=n_neighbors)
    umapped_data = reducer.fit_transform(data_for_umap)
    return umapped_data

def grouping_testing(use_cuda):


    parser = argparse.ArgumentParser(description='Segmentation algorithm settings')

    parser.add_argument('--dset_dir', default='datasets', type=str, help='dataset directory')
    
    parser.add_argument('--training_result_dir', default='results', type=str, help='Where model outputs are stored.')
    parser.add_argument('--dataset', default='dSwissFlag', type=str, help='dataset name.')
    parser.add_argument('--training_name', default='dSwissFlag_VAE_1', type=str, help='the name referring to a specific training configuration.')
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str,  
                        help='Load previous checkpoint - insert checkpoint file name.')
    parser.add_argument('--model', default = 'BetaVAE_dstripes', type=str,\
                        help = 'name of model you hope to instantiate from the checkpoint.')
    parser.add_argument('--require_UMAP', action='store_true')
    parser.add_argument('--UMAP_html_plot', action='store_true')
    parser.add_argument('--require_recon_plot', action='store_true')
    parser.add_argument('--UMAP_n_neighbors', default=400, type=int, help="The n_neighbors field of UMAP algorithm.")
    parser.add_argument('--UMAP_num', default=2, type=int, help='Number of test samples you hope to perform UMAP on.')
    parser.add_argument('--image_size', default=64, type=int, help='Size of the input image sides.')
    parser.add_argument('--num_workers', default=1, type=int, help='dataloader num_workers')

    parser.add_argument('--nc', default=3, type=int, help='Number of channels for this model test.')
    parser.add_argument('--z_dim', default=32, type=int,
                        help='Number of latent variables used for this model in this experiment.')

    parser.add_argument('--testing_result_dir', default='segmentation_testing', type=str,
                        help='A folder used to contain all the segmentation results.')
    parser.add_argument('--noise_sample_distribution', default='gaussian', type=str,
                        help='The distribution used for noise samples in the latent layer.')
    parser.add_argument('--noise_sample_std', default=0.0001, type=float,
                        help='The std of the latent-perturbing noise.')
    parser.add_argument('--perturb_num', default=70, type=int,
                        help='The number of perturbations on latent nodes.')

    parser.add_argument('--n_clusters', default=-1, type=int, 
                        help='The number of clusters you which to acquire from the clustering method. If set to be non-positive, then the program will automatrically choose the cluster number, based on the different numbers of values in the mask.')
    parser.add_argument('--test_num', default=10, type=int, 
                        help='The number of test samples you hope to test upon.')


    args, _ = parser.parse_known_args()

    clustering_args = {'algorithm': 'AgglomerativeClustering',
                       'n_clusters': args.n_clusters,
                       'metric': 'euclidean',
                       'linkage': 'ward'}

    print(f"testing on {args.training_name}: noise_std: {args.noise_sample_std}, perturb_num: {args.perturb_num}")

    parent_directory = os.path.join('.', args.training_result_dir, args.dataset, args.training_name)
    checkpoint_path = os.path.join(parent_directory,  args.ckpt_dir, args.ckpt_name)
    n_clusters_name = args.n_clusters if args.n_clusters > 0 else "auto"
    save_directory = os.path.join(parent_directory,  args.testing_result_dir, f"n_clusters_{n_clusters_name}_noise_std_{args.noise_sample_std}_perturb_num_{args.perturb_num}")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = cuda(eval(args.model)(z_dim=args.z_dim, nc=args.nc), use_cuda)  
    model.load_state_dict(checkpoint['model_states']['net'])
    segmenter = SegmentationTester(args=args, model=model, save_directory=save_directory,
                                    clustering_args=clustering_args, cuda=use_cuda)
    segmenter.run_segmentation_tests(args.test_num)




def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    use_cuda = torch.cuda.is_available()
    grouping_testing(use_cuda)


if __name__ == '__main__':
    main()
