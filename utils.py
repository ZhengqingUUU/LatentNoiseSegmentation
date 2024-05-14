import os
import sys
import datetime
import argparse
import glob

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor


def str2bool(v):
    # from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def fprint(text, file, timestamp=False):
    """Print text to screen and write to FPRINT_FILE."""
    # Add timestamp if requested
    if timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = timestamp + ' ' + text
    # Redirect stdout to file
    original = sys.stdout
    if file is not None:
        sys.stdout = open(file, 'a+')
        print(text)
        # Set stdout back to original
        sys.stdout = original
    print(text)


def get_latest_version_folder(training_outcome_savedir):
    """among all the folders with the name training_outcome_savedir(_vX) with X being a number,
    find the latest version and return it 
    """
    outcome_folders = glob.glob(os.path.join(training_outcome_savedir+r"*",))
    original_length = len(training_outcome_savedir)
    _v_length = len("_v")
    total_length = original_length + _v_length
    max_version_num = -1
    for folder in outcome_folders:
        try: version_num = int(folder[total_length:])
        except ValueError: version_num = 0
        max_version_num = max(max_version_num, version_num)
    if max_version_num == 0: pass
    else: training_outcome_savedir = training_outcome_savedir+"_v"+str(max_version_num)
    return training_outcome_savedir

def versioning_folder(training_outcome_savedir):
    """If there exists a training folder with the same name, 
    append a versioning string to the end of the folder, like _v1, _v2, _v3 ...
    """
    outcome_folders = glob.glob(os.path.join(training_outcome_savedir+r"*",))
    if len(outcome_folders) == 0: return training_outcome_savedir
    else: # search for the latest version and increase the versioning number by 1
        original_length = len(training_outcome_savedir)
        _v_length = len("_v")
        total_length = original_length + _v_length
        max_version_num = -1
        for folder in outcome_folders:
            try: version_num = int(folder[total_length:])
            except ValueError: version_num = 0
            max_version_num = max(max_version_num, version_num)
        if max_version_num == 0: training_outcome_savedir += '_v1'
        else: training_outcome_savedir = training_outcome_savedir+"_v"+str(max_version_num+1)
        return training_outcome_savedir

def plot_reconstruction(input_image: Tensor, reconstruction: Tensor, savedir: str, image_name: str):
    # Add sigmoid, since it's computed in the loss function (but not in the model itself)
    recon = torch.sigmoid(reconstruction).cpu().detach().numpy()
    input_image = input_image.cpu().detach().numpy()
    # Shuffle channel axis to last for imshow
    recon = np.einsum('kij->ijk', recon)
    input_image = np.einsum('kij->ijk', input_image)
    img = np.hstack((input_image, recon))
    img = np.squeeze(img) # for compatibility with images with only one channel
    plt.imshow(img)
    plt.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    os.makedirs(savedir, exist_ok=True)
    plt.savefig(os.path.join(savedir, image_name+'_recon.jpg'))
    plt.close()

def show_images_grid(imgs_, num_images=25, ):
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[ax_i],)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')