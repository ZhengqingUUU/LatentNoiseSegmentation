# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2021 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================
# This code has been adapted by Zhengqing Wu

import torch
from utils import cuda
from torch.autograd import Variable

class GECO():

    def __init__(self, goal, step_size, alpha=0.99, beta_init=1.0,
                 beta_min=1e-10, compute_on_cuda = True, speedup=None):
        self.err_ema = None
        self.goal = goal
        self.step_size = step_size
        self.alpha = alpha
        self.beta = Variable(cuda(torch.tensor(beta_init),uses_cuda=compute_on_cuda))
        self.beta_min = Variable(cuda(torch.tensor(beta_min),uses_cuda=compute_on_cuda))
        self.beta_max = Variable(cuda(torch.tensor(1e10),uses_cuda=compute_on_cuda))
        self.speedup = speedup

    def to_cuda(self):
        self.beta = self.beta.cuda()
        if self.err_ema is not None:
            self.err_ema = self.err_ema.cuda()

    def loss(self, err, kld, compute_on_cuda = True,  freeze_beta = False):
        # Compute loss with current beta
        loss = err + self.beta * kld
        # Update beta without computing / backproping gradients
        with torch.no_grad():
            if self.err_ema is None:
                self.err_ema = err
            else:
                self.err_ema = (1.0-self.alpha)*err + self.alpha*self.err_ema
            if not freeze_beta: 
                # freeze_beta is True when we do not want to updata beta, e.g. during validation
                constraint = (self.goal - self.err_ema)
                if self.speedup is not None and constraint.item() > 0:
                    factor = Variable(cuda(torch.exp(self.speedup * self.step_size * constraint),uses_cuda = compute_on_cuda))
                else:
                    factor = Variable(cuda(torch.exp(self.step_size * constraint),uses_cuda = compute_on_cuda))
                self.beta = (factor * self.beta).clamp(self.beta_min, self.beta_max)
        # Return loss
        return loss
