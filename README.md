# Insightface_EfficientNet
# Intro
This repo implements the Deep Face Recognition part of Insightface(github) with a backbone of EfficientNet(github). 
# Training strategies and results
Training data is MS1M, face detect by MTCNN and resized to 112x112.  
a.EfficientNet(b0) with batchsize 80 + Argface(m=64,s=0.5) + focalloss(gam=2)  
The results is trained 50 epoch, pretrined model can download in here.
| LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
| ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
| 0.9955 | 0.9940   | 0.9347   | 0.9545      | 0.9532  | 0.8973  | 0.9320    |  

b.other case is updating...
