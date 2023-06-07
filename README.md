# PGD-UNet
## Requirements
* python==3.8 
* numpy==1.23
* pytorch==1.13   
* scikit-learn==1.2.1 

## How to run
- train.py 用于训练原始的UNet网络
- predict.py 用于输出测试集的预测结果，以及对应的dice指标
- PGD.py 用于生成PGD adversarial example

