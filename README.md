# StSqDB: A Video Database for Figure Skating Step Sequencing


## Getting Started
1 anno_data.py
 * imageとlavel(0~12)の情報が入っているpklファイルの作成

2 preprocess.py
 * DataAugmentationを行う  
  * リサイズ(224*224),反転,色補正  
  * 300フレームごとに一つの動画にする  
  * 交差検証のために4つにsplit(pklファイルの作成)


### Train
bs=8  
iteration=3000  


### Evaluate
bs=1(しかできない。。なぜ？？）


### これまでの精度
iteration=2000, bs=1, seq_length=300  
 * Average PCE: 0.02068034188034188
 


