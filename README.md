# StSqDB: A Video Database for Figure Skating Step Sequencing


## Getting Started
1 anno_data.py
 * imageとlavel(0~12)の情報が入っているpklファイルの作成

2 preprocess.py
 * DataAugmentationを行う  
  * リサイズ(224*224),反転,色補正  
  * 300フレームごとに一つの動画にする  
  * 交差検証のために4つにsplit(pklファイルの作成)


## Train
bs=8  
iteration=3000  


## Evaluate
bs=1(しかできない。。なぜ？？）


## これまでの精度
iteration=2000, bs=1, 動画を切り分けない
 * Average PCE: 0.03138888888888889

iteration=2000, bs=1, frame=300で動画を切り分け 
 * Average PCE: 0.02068034188034188
 
 ## 今やってること
 ### pallas
 iteration=2000, bs=8, frame=300
 ### glacus
 iteration=6000, bs=16, frame=100


