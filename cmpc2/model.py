import numpy as np 
import torch
import webrtcvad
from PIL import Image
from .mfcc import MFCC,get_fbank
from .fv_wrapper import  FV_Wrapper,load_model



def load(path):
    # load ckpt
    # model load chechpoints
    fv = FV_Wrapper(pretrain=False, last_dim=512, proj_dim=512, proj=False, test_only=False)
    model = load_model(fv, path)
    return model

def audio_preprocess(valid_level=2,min_time=800):
    def _wav2fbank(wav_file):
        vad_obj = webrtcvad.Vad(valid_level)
        mfc_obj = MFCC(nfilt=64, lowerf=20., upperf=7200., samprate=16000, nfft=1024, wlen=0.025)
        fbank = get_fbank(wav_file,vad_obj, mfc_obj, min_time)
        return torch.from_numpy(fbank)
    return _wav2fbank

def load_voice(min_time=800):
    def _load(np_path):
        voice_data = np.load(np_path)#n_frames,64 ?
        
        voice_data = voice_data.T.astype('float32') 
        pt = np.random.randint(voice_data.shape[1] - min_time + 1)
        voice_data = voice_data[np.newaxis,:,pt:pt+min_time]
        
        return torch.from_numpy(voice_data)
    return _load

def visual_preprocess(img_size=224):
    def _load_face(file):
        if isinstance(file, Image):
            face = file.convert('RGB').resize([img_size, img_size])
        else:
            face = Image.open(file).convert('RGB').resize([img_size, img_size])
        face = np.transpose(np.array(face), (2, 0, 1))
        face = ((face - 127.5) / 127.5).astype('float32')
        return torch.from_numpy(face)
    return _load_face
