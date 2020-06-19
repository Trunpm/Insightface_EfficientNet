# Insightface_EfficientNet
Pytorch implements the Deep Face Recognition part of Insightface([github](https://github.com/deepinsight/insightface))  with a backbone of EfficientNet([github](https://github.com/lukemelas/EfficientNet-PyTorch)). 
# About EfficientNet
Official explanation: EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models. We develop EfficientNets based on AutoML and Compound Scaling. In particular, we first use [AutoML Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) to develop a mobile-size baseline network, named as EfficientNet-B0; Then, we use the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to B7.
<table border="0">
<tr>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/params.png" width="100%" />
    </td>
    <td>
    <img src="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/flops.png", width="90%" />
    </td>
</tr>
</table>

Details about the EfficientNet models are below: 
|    *Name*         |*# Params*|*Top-1 Acc.*|
|:-----------------:|:--------:|:----------:|
| `efficientnet-b0` |   5.3M   |    76.3    |
| `efficientnet-b1` |   7.8M   |    78.8    |
| `efficientnet-b2` |   9.2M   |    79.8    |
| `efficientnet-b3` |    12M   |    81.1    |
| `efficientnet-b4` |    19M   |    82.6    |
| `efficientnet-b5` |    30M   |    83.3    |
| `efficientnet-b6` |    43M   |    84.0    |
| `efficientnet-b7` |    66M   |    84.4    |

# Data Preparation for face recognition
downloading the Training data [MS1M](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo), face is detected by MTCNN and resized to 112x112. If you need to tansfer the `.bin` or `.rec` files into images(.jpg),please run the script `python GetImages.py` under your data fold, note that maxnet should be install.
# Training strategies and results  
a. EfficientNet(b0,Params is 5.3M) with batchsize 80 + Argface(m=64,s=0.5) + focalloss(gam=2)  
The results is trained 50 epoch, pretrained model can be download in [here]()(few days late...).  
| LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
| ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
| 0.9955 | 0.9940   | 0.9347   | 0.9545      | 0.9532  | 0.8973  | 0.9320    |  

b. other pretrained model b1, b2, ..., b7 and results is updating...
# PS
If you have questions, post them as GitHub issues.
