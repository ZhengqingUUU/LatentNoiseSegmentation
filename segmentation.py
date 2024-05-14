
from typing import Dict, Union, Optional, Callable
from numpy import ndarray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.cluster import AgglomerativeClustering


class NoisySegmentation(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 perturb_num: int,
                 clustering_args: Dict[str, Union[int, str]],
                 device: str,
                 image_size: int = 64,
                 noise_sample_distribution: str = 'gaussian',
                 noise_sample_std: float = 0.001
                 ):
        """A class that stipulates the process of segmentation."""
        super().__init__()
        # Model and noise inits
        self.model = model
        self.perturb_num = perturb_num 
        self.image_size = image_size
        self.noise_sample_distribution = noise_sample_distribution
        if self.noise_sample_distribution == 'uniform':
            print('Note: noise_sample_std for the uniform distribution denotes the range of the'
                  ' uniform distribution, not its standard deviation.')
        self.noise_sample_std = noise_sample_std

        # Clustering algorithm init
        self.clustering_algorithm = self.parse_clustering_args(clustering_args)

        self.device = device

    def forward(self, x: Tensor, right_n_clusters:int) -> ndarray:
        """
        perform perturbation and clustering. return a (image_size,image_size)-shaped numpy array
        containing the inferred segmentation masks.
        right_n_clusters will be used as the n_clusters field of the clustering algorithm if n_clusters is non-positive.
        """
        # Initialize tensor for model outputs
        subtracted_outputs = torch.zeros(size=(self.perturb_num - 1, self.model.nc,
                                               self.image_size, self.image_size)) 
        latents_unperturbed = self.get_latent_means(x)
        reconstruction = torch.sigmoid(self.model.decoder(latents_unperturbed))

        # noise perturbation loop
        for step in range(self.perturb_num):
            latents = latents_unperturbed+self.noise_sample()  
            output = torch.sigmoid(self.model.decoder(latents))
            if step > 0:
                subtracted_outputs[step - 1, :, :, :] = previous_output - output
            previous_output = output
        # normalization for a given pixel 
        normalized_outputs = F.normalize(subtracted_outputs,dim=0)
        # for each pixel, concatenate the subtraction values from three channels together
        channel_stacked_outputs = normalized_outputs.reshape(-1,self.image_size, self.image_size).detach().cpu().numpy()
        prediction = self.cluster(channel_stacked_outputs, right_n_clusters=right_n_clusters)
        return reconstruction, prediction, channel_stacked_outputs 

    def get_latent_means(self, x: Tensor) -> Tensor:
        return self.model.encoder(x)[:, :self.model.z_dim]

    def noise_sample(self) -> Tensor:
        means = torch.zeros(self.model.z_dim)

        if self.noise_sample_distribution == 'gaussian':
            noise = torch.normal(mean=means, std=self.noise_sample_std).to(self.device)
            return noise

        else:
            raise NotImplementedError

    def cluster(self, channel_stacked_outputs: Tensor, right_n_clusters: int) -> ndarray:
        """
        channel_stacked_outputs: Tensor
            has shape (batch_size, (perturb_num - 1) * nc, image_size, image_size)
        This function with cluster the above input variable
        """
        data_to_be_clustered = channel_stacked_outputs.reshape(-1, self.image_size**2)
        if self.clustering_algorithm.n_clusters <= 0: # implies that no fixed number of clusters is provided, this program will use the "ground-truth" number of clusters indicated by masks (number of different values existing in the masks.)
            original_n_clusters = self.clustering_algorithm.n_clusters
            self.clustering_algorithm.n_clusters = right_n_clusters
        self.clustering_algorithm.fit(data_to_be_clustered.T)
        flattened_labels = self.clustering_algorithm.labels_
        labels = flattened_labels.reshape(self.image_size, self.image_size)

        try: #original_n_cluster does not exist if the if-clause above is not entered.
            self.clustering_algorithm.n_clusters = original_n_clusters # change it to negative so that it will be switch to a potentially different right_n_clusters in the next round of clustering.
        except: pass  
        
        return labels

    def parse_clustering_args(self,
                              clustering_args: Dict[str, Union[int, str]]
                              ) -> Dict[str, Union[int, Callable]]:
        """A function to select the correct settings for clustering, given clustering_args"""
        # Initialize clustering dict for ease of readability

        # Determine clustering algorithm
        if clustering_args['algorithm'] == 'AgglomerativeClustering':
                clustering_algorithm =  AgglomerativeClustering(n_clusters=clustering_args['n_clusters'], # Could be 0 or negative.
                                                                affinity=clustering_args['metric'],
                                                                linkage=clustering_args['linkage'])
        else:
            raise NotImplementedError
        return clustering_algorithm



