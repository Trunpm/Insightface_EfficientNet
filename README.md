# Insightface_EfficientNet
# Intro
This repo implements the Deep Face Recognition part of Insightface([github](https://github.com/deepinsight/insightface))  with a backbone of EfficientNet([github](https://github.com/lukemelas/EfficientNet-PyTorch)). 
# Training strategies and results
Training data is [MS1M](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo), face detect by MTCNN and resized to 112x112.  
a. EfficientNet(b0,Params is 5.3M) with batchsize 80 + Argface(m=64,s=0.5) + focalloss(gam=2)  
The results is trained 50 epoch, pretrained model can be download in [here]()(few days late).  
| LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
| ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
| 0.9955 | 0.9940   | 0.9347   | 0.9545      | 0.9532  | 0.8973  | 0.9320    |  

b. other pretrained model [b1](), [b2](), ..., [b7]() and results is updating...
# How to use
# PS
If you have questions, post them as GitHub issues.
