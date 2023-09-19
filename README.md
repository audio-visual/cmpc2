# cmpc2
 face and speech contrastive model

please refer to the original repository: https://github.com/Cocoxili/CMPC.git  


this repository convert the cmpc to a python package, in order to get the features more convenient like CLIP.

## How to use:  
1. install it  
`pip install git+https://github.com/audio-visual/cmpc2.git` 

2. get features 
```python
import cmpc2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = 'xx.tar' #pretrained cmpc weight
model = cmpc2.load(path).to(device)
# you should manually change the model mode
model.eval()

wav_path = 'xx.wav'
img_size = 224
frame_path = 'xx.jpg/png'

visual_preprocess, audio_preprocess = cmpc2.visual_preprocess(img_size), cav_mae.audio_preprocess()
audio = visual_preprocess(wav_path).unsqueeze(0).to(device) #(1,1,64,800)
visual = visual_preprocess(frame_path).unsqueeze(0).to(device) #(1,3,224,224)

audio_emb, frame_emb = model(audio, visual)

```

3. get other modal samples  
```python
memory_path = '/root/projects/CMPC/checkpoints/origin/cmpc/frame_memory_best.pth.tar'
df_path = '/root/projects/CMPC/data/training.csv'

image_paths = getKNearestSamples(audio_emb,memory_path,df_path, cluster=1000,K=3)
```

