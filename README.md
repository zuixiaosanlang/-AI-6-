
#准备
# 下载数据，按结构放置
所有数据均在此了
链接：https://pan.baidu.com/s/1TC9GTDgyKnDBf07sHp8gtg 
提取码：asdf

## **Training**
The data folders should be:
```
Dataset
    * inpainting
        - train_A # 手写 images
        - train_B # masks ，abs(images - gts) > 10 产生
        - train_C # gts

    *result      # 预测结果图片保存路径
    *test_dataaet # 待测试数据集
```

## **脚本说明**
```
1）infer.py
    手写擦除testB采用该脚本生成，存在动态shape。对Dataset/test_dataset中的图片推理结果，保存于Dataset/result文件夹
    python infer.py

2）ckpt2pb.py
    由tf快照生成.pb模型脚本, .pb保存于pd_model中
    python ckpt2pb.py
    
3）x2paddle_code.py
    该脚本是由paddle工具转换.pb模型来的,用于paddle推理，paddle框架推理用。
    对Dataset/test_dataset中的图片推理结果，保存于Dataset/result文件夹
    python x2paddle_code.py 

4）train.py
    训练脚本。模型保存于logs/pre-trained；同时训练过程中测试效果图亦保存于此文件夹
    python train.py 

```

This repo base on  the code:
    https://github.com/vinthony/ghost-free-shadow-removal
    与该仓库去除阴影原理一样