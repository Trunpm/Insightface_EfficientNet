# Insightface_EfficientNet
# Intro
This repo implements the Deep Face Recognition part of Insightface(github) with a backbone of EfficientNet(github). 
# Training strategies and results
Training data is MS1M, face detect by MTCNN and resized to 112x112.  
a.EfficientNet(b0) with batchsize 80 + Argface(m=64,s=0.5) + focalloss(gam=2)  
b.other case is updating...
