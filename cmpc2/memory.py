# import numpy as np 
import torch
import pandas as pd
import os

from torch.nn import functional as F
# from collections import Counter

def getKNearestSamples(audio_feature, memory_path, df_path, cluster=1000,K=3):
    # in the original cmpc, the cluster set: [500,1000,1500]
    cluster_index = cluster//500 -1
    # load memory
    mem = torch.load(memory_path)
    df = pd.read_csv(df_path, sep=',')
    centroids = mem['centroids']
    centroid = centroids[cluster_index]
   
    audio_feature = F.normalize(audio_feature)
    centroid = F.normalize(centroid, dim=-1)
    D_ij = ((audio_feature - centroid) ** 2).sum(-1)
    
    # choose K nearest centroids
    ids = torch.topk(-D_ij,K).indices
   
    inst2clusters = mem['inst2cluster']
    inst2cluster = inst2clusters[cluster_index]
    
    inst2cluster = list(inst2cluster.detach().cpu().numpy())
   
    image_paths = []
    for inst,cluster in enumerate(inst2cluster):
        if cluster in ids:
            image_paths.append(os.path.join(df['VGGFace1 ID'][inst],'1.6',df['video_id'][inst]))
  
    return image_paths

def getKNearestSamples_bak(audio_feature, memory_path, df_path, cluster=1000,K=3):
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
    import cmpc2
    memory_path = '/home/cwy/下载/frame_memory_best.pth.tar'
    df_path = '/home/cwy/下载/training.csv'
    audio_path = './example/id10840_0eRPKDAV-I0_00001.wav'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_preprocess = cmpc2.audio_preprocess()
    audio = audio_preprocess(audio_path).unsqueeze(0).to(device)
    path = '/home/cwy/下载/model_best.pth.tar' #pretrained cmpc weight
    model = cmpc2.load(path).to(device)
    # you should manually change the model mode
    model.eval()

    frame = torch.randn(1,3,224,224).to(device)
    audio_emb, frame_emb = model(audio,frame)
    image_paths = getKNearestSamples(audio_emb,memory_path,df_path)
    print(image_paths)



