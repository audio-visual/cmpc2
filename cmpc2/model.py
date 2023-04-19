import numpy as np 
import torch
from .fv_wrapper import  FV_Wrapper,load_model



def load(path):
    # load ckpt
    # model load chechpoints
    fv = FV_Wrapper(pretrain=False, last_dim=512, proj_dim=512, proj=False, test_only=False)
    model = load_model(fv, path)
    return model