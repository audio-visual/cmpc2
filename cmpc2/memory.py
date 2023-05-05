# import numpy as np 
import torch
import pandas as pd
import os
# import cmpc2
from torch.nn import functional as F
# from collections import Counter

def getKNearestSamples(audio_feature, memory_path, df_path, cluster=1000,K=3):
    cluster_index = cluster//500 -1
    # load memory
    mem = torch.load(memory_path)
    df = pd.read_csv(df_path, sep=',')
    centroids = mem['centroids']
    centroid_1000 = centroids[cluster_index]
    # print(centroid_1000.shape)
    # audio_feature:(1,512)
    # print(audio_feature.shape)
    audio_feature = F.normalize(audio_feature)
    centroid_1000 = F.normalize(centroid_1000, dim=-1)
    D_ij = ((audio_feature - centroid_1000) ** 2).sum(-1)
    # print(torch.topk(-D_ij,K))
    ids = torch.topk(-D_ij,K).indices
    # print(ids)
    
    inst2clusters = mem['inst2cluster']
    inst2cluster_1000 = inst2clusters[cluster_index]
    # print(inst2cluster_1000.shape)
    inst2cluster_1000 = list(inst2cluster_1000.detach().cpu().numpy())
    
    # x= Counter(inst2cluster_1000)
    # print(x.most_common(10))
    # print(Counter(inst2cluster_1000))

   
    image_paths = []
    for inst,cluster in enumerate(inst2cluster_1000):
        if cluster in ids:
            image_paths.append(os.path.join(df['VGGFace1 ID'][inst],'1.6',df['video_id'][inst]))
        # centroids to video index        
   
    return image_paths

if __name__ == '__main__':
    memory_path = '/root/projects/CMPC/checkpoints/origin/cmpc/frame_memory_best.pth.tar'
    df_path = '/root/projects/CMPC/data/training.csv'
    # audio_feature = torch.randn(1,512).cuda() # TODO change to the right training audio sample!
    audio_path = '/hy-tmp/fbank/fbank/id10840/0eRPKDAV-I0/00001.npy'
    audio = cmpc2.load_voice(audio_path)
    audio = audio[None,...]
    path = '/root/projects/CMPC/checkpoints/origin/cmpc/model_best.pth.tar' #pretrained cmpc weight
    model = cmpc2.load(path)
    frame = torch.randn(1,3,128,128)
    audio_emb, frame_emb = model(audio.cuda(),frame.cuda())
    image_paths = getKNearestSamples(audio_emb,memory_path,df_path)
    print(image_paths)



