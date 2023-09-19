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

visual_preprocess, audio_preprocess = cmpc2.visual_preprocess(img_size), cmpc2.audio_preprocess()
audio = audio_preprocess(wav_path).unsqueeze(0).to(device) #(1,1,64,800)
visual = visual_preprocess(frame_path).unsqueeze(0).to(device) #(1,3,224,224)
# or (to keep the same with CLIP)
# visual = visual_preprocess(Image.open(frame_path)).unsqueeze(0).to(device)

audio_emb, frame_emb = model(audio, visual) # (1,512)

```