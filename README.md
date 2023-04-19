# cmpc2
 face and speech contrastive model

please refer to the original repository: https://github.com/Cocoxili/CMPC.git  


this repository convert the cmpc to a python package, in order to get the features more convenient like CLIP.

## How to use:  
1. install it  
`pip install git+https://github.com/wuyangchen97/cmpc2.git` 
2. import cmpc and use 
```python
import cmpc2
# (batch_size, channel, features, time)
audio = torch.randn(2,1,64,800)
# (batch_size, channel, img_size, img_size)
frame = torch.randn(2,3,128,128)
path = 'xx.tar' #pretrained cmpc weight
model = cmpc2.load(path)

# audio_emb,frame_emb:(batch_size, 512)
audio_emb, frame_emb = model(audio,frame)

```