# cmpc2
 face and speech contrastive model

please refer to the original repository: https://github.com/Cocoxili/CMPC.git  


this repository convert the cmpc to a python package, in order to get the features more convenient like CLIP.

## How to use:  
1. install it  
`pip install git+https://github.com/wuyangchen97/cmpc2.git` 

2. load voice from wav file or processed numpy file(assume already normalized)  
```python
import cmpc2
wav_path = 'xx.wav'
audio = cmpc2.tokenize(wav_path) #(1,64,800)
# or directly use np file
# np_path = 'xx.np'
# audio = cmpc2.load_voice(np_path)
```
3. load image
```python
frame_path = 'xx.jpg/png'
img_size = 128
frame = cmpc2.load_face(frame_path,img_size)
```

4. get features  
```python
# (batch_size, channel, features, time)
# audio = torch.randn(2,1,64,800)
audio = audio[None,...]
# (batch_size, channel, img_size, img_size)
# frame = torch.randn(2,3,128,128)
frame = frame[None,...]
path = 'xx.tar' #pretrained cmpc weight
model = cmpc2.load(path)

# audio_emb,frame_emb:(batch_size, 512)
audio_emb, frame_emb = model(audio.cuda(),frame.cuda())

```

5. get other modal samples  
```python
memory_path = '/root/projects/CMPC/checkpoints/origin/cmpc/frame_memory_best.pth.tar'
df_path = '/root/projects/CMPC/data/training.csv'

image_paths = getKNearestSamples(audio_emb,memory_path,df_path, cluster=1000,K=3)
```

