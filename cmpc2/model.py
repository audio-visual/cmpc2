import numpy as np 
import torch
from .fv_wrapper import  FV_Wrapper



def load(path):
    # load ckpt
    fv = FV_Wrapper(pretrain=False, last_dim=512, proj_dim=512, proj=False, test_only=False)
    return fv.get_models()