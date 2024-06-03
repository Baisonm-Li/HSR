# 高光谱超分辨模型

## 运行环境

主要环境
Python 3.9.18
pytorch 2.1.1+cu121
numpy 1.25.2
在项目目录（requirements.txt同级目录）下直接在终端运行
```
pip install -r requirements.txt
```

可以直接导入所需依赖

## 数据下载

CAVE: http://www1.cs.columbia.edu/CAVE/databases/multispectral/
Havard: https://vision.seas.harvard.edu/hyperspec/download.html
PaviaU: http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
Urban: http://www.erdc.usace.army.mil/Media/Fact-Sheets/Fact-Sheet-Article-View/Article/610433/hypercube/#
Chikusei: https://naotoyokoya.com/Download.html

## 数据预处理
./datasets/shells 目录中有对应数据集的不同处理方式

## 目录介绍

```
datasets -- 训练数据和测试数据
models -- 模型的实现
data -- 数据类的实现
plot.ipynb -- 展示模型生成图片
main.py -- 模型的主入口
util -- 工具函数集合
```

## 训练命令

`python main --model_name 'models文件夹里面的模型名' --dataset '数据名称'`

## 实例

`python main --model_name SGANet --dataset Urban`
就是使用SGANet模型在Urban数据集上进行训练

## 其他参数

```
parse.add_argument('--model_name',type=str) # 模型名
parse.add_argument('--dataset',type=str) # 数据集
parse.add_argument('--check_point',type=str,default=None) # 模型继续训练，这里是模型的路径
parse.add_argument('--lr',type=int,default=4e-4) # 学习率
parse.add_argument('--batch_size',type=int,default=32) 
parse.add_argument('--epochs',type=int,default=1000)
```
