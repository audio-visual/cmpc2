import torch
import torch.nn as nn
import torch.nn.functional as F
from .fv_resnet import *


class Head(nn.Module):
    def __init__(self, last_dim, proj_dims):
        super(Head, self).__init__()

        projection = [nn.Dropout(0.5), nn.Linear(last_dim, proj_dims)]

        self.projection = nn.Sequential(*projection)
        self.out_dim = proj_dims

    def forward(self, x):
        return self.projection(x)

def load_weights_from_cmpc(model, pretrained_dict):# load single model
   
    model_dict = model.state_dict()
    # print('keys in init model dict:')
    # for k,v in model_dict.items():
    #     print(k)
    # print('voice_subnet.bn1.bias',model.voice_subnet.bn1.bias)
    # print()
    # pretrained_dict = torch.load(path, map_location={'cuda:0': 'cpu'})
    # pretrained_dict.map_location{} TODO 
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict['model'].items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    # print('keys in filtered pretrained dict:')
    # for k,v in pretrained_dict.items():
    #     print(k)
    # print('voice_subnet.bn1.bias',pretrained_dict['voice_subnet.bn1.bias'])
    # print()
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print('loaded cmpc ckp ')
    # return model # 可以不返回，直接修改了model里面的weight

def load_model(model, checkpoint_fn): # load face-voice model
    # model = models.__dict__[cfg['model']['arch']](**cfg['model']['args'])
    print("loads chechpoint for both model:",checkpoint_fn)
    ckp = torch.load(checkpoint_fn, map_location='cpu')
    # model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
    model.load_state_dict(ckp['model'])
    # logger.info('Load from {} at iteration #{}'.format(checkpoint_fn, ckp['epoch']))
    # model = model.cuda()
    # model.eval()

    return model


class FV_Wrapper(nn.Module):
    def __init__(self, pretrain, last_dim, proj_dim, proj, test_only=False):
        super(FV_Wrapper, self).__init__()
        self.test_only = test_only
        self.proj = proj
        self.voice_subnet = resnet34(input_channel=1, out_dim=last_dim, pretrain=pretrain)
        self.face_subnet = resnet34(input_channel=3, out_dim=last_dim, pretrain=pretrain)

        # self.projnet = Head(last_dim=last_dim, proj_dims=proj_dim)

        # self.voice_subnet_proj = nn.Sequential(self.voice_subnet, self.projnet)
        # self.face_subnet_proj = nn.Sequential(self.face_subnet, self.projnet)
        # self.test_only = test_only
    # def get_models(self,path):
    #     return self.voice_subnet, self.face_subnet

    def forward(self, audio, frame):
        audio_emb = self.voice_subnet(audio)
        # if self.proj:
        #     audio_emb = self.projnet(audio_emb)
        frame_emb = self.face_subnet(frame)
        # if self.proj:
        #     frame_emb = self.projnet(frame_emb)
        return audio_emb, frame_emb