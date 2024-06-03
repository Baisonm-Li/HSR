import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset,DataLoader
import os
import logging
### 导入模型
from models.SSRNet import SSRNet
from models.ResTFNet import ResTFNet
from models.HSRNet import HSRNet
from models.FuseFormer import FuseFormer
from models.SGANet import SGANet
from models.MHFNet import MHFNet
from models.CSSNet import CSSNet
import logging
from data import H5Dataset,NPZDataset
import numpy as np
from utils import Metric,get_model_size,test_speed,beijing_time, set_logger
import argparse
parse = argparse.ArgumentParser()

# 这里是在命令行中可以输入的参数，不输入则使用defalt指定的值
parse.add_argument('--log_out',type=int,default=1) # 这里是否使用日志文件 1 表示使用， 0 表示不使用
parse.add_argument('--model_name',type=str) # 这里指定需要运行的模型名字
parse.add_argument('--dataset',type=str) # 这里指定训练的数据集
parse.add_argument('--check_point',type=str,default=None) # 这里指定预训练的模型，.pth文件继续训练
parse.add_argument('--check_step',type=int,default=50) # 这里指定多少个epoch保存一次模型
parse.add_argument('--lr',type=int,default=4e-4) # 指定学习率
parse.add_argument('--batch_size',type=int,default=32) # 
parse.add_argument('--epochs',type=int,default=1000) # 运行的轮次
args = parse.parse_args()

# 确定运行设备 GPU 或者 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = args.model_name
model = None
HSI_bands = 31
test_dataset_path = None
train_dataset_path = None

# 加载数据集，对于不同的数据集设置的波段不同
if args.dataset == 'CAVE':
    test_dataset_path = './datasets/CAVE_test.npz'
    train_dataset_path = './datasets/CAVE_train.npz'
    HSI_bands = 31 # 指定波段
if args.dataset == "PaviaU":
    test_dataset_path = './datasets/PaviaU_test.npz'
    train_dataset_path = './datasets/PaviaU_train.npz'
    HSI_bands = 103
if args.dataset == "Havard":
    test_dataset_path = './datasets/Havard_test.npz'
    train_dataset_path = './datasets/Havard_train.npz'
    HSI_bands = 31
if args.dataset == "Urban":
    test_dataset_path = './datasets/Urban_test.npz'
    train_dataset_path = './datasets/Urban_train.npz'
    HSI_bands = 162
if args.dataset == "Chikusei":
    test_dataset_path = './datasets/Chikusei_test.npz'
    train_dataset_path = './datasets/Chikusei_train.npz'
    HSI_bands = 31
    
# 加载指定的数据集
if model_name.startswith('SGANet'):
    model = SGANet(HSI_bands,hidden_feature_dim=64,group_num=8)
if model_name.startswith('SSRNet'):
    model = SSRNet(HSI_bands,)
if model_name.startswith('HSRNet'):
    model = HSRNet(HSI_bands)
if model_name.startswith('FuseFormer'):
    model = FuseFormer(HSI_bands)
if model_name.startswith('MHFNet'):
    model = MHFNet(HSI_bands)
if model_name.startswith('CSSNet'):
    model = CSSNet(HSI_bands)
if model_name.startswith('ResTFNet'):
    model = ResTFNet(HSI_bands)
# 导入模型
model = model.to(device)
# 指定日志文件输出的路径
if args.check_point is not None:
    model.load_state_dict(torch.load(args.check_point))
    print(f'check_point: {args.check_point}')
log_dir = f'./trained_models/{model_name},{args.dataset},{beijing_time()}'
if not os.path.exists(log_dir) and args.log_out == 1:
    os.mkdir(log_dir)
    
logger = set_logger(model_name,log_dir,args.log_out)
model_size = get_model_size(model)
# 测试模型的参数和运行速度
inference_time,flops,params = test_speed(model,device,HSI_bands)
logger.info(f'[model:{args.model_name},dataset:{args.dataset}],model_size:{params},inference_time:{inference_time:.6f}S,FLOPs:{flops}')
# 损失函数
loss_func = torch.nn.L1Loss()
optimizer = torch.optim.Adam(lr=args.lr,params=model.parameters())
# 学习率衰减策略
StepLR(optimizer=optimizer,step_size=100,gamma=0.1)
# 加载数据集
test_dataset = NPZDataset(test_dataset_path)
train_dataset = NPZDataset(train_dataset_path)
train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size * 4)

# 训练
def train():
    model.train()
    loss_list = []
    for epoch in range(0, args.epochs):
        for idx,loader_data in enumerate(train_dataloader):
            GT,LRHSI,RGB = loader_data[0].to(device),loader_data[1].to(device),loader_data[2].to(device)
            optimizer.zero_grad() # 梯度清零
            preHSI = model(LRHSI,RGB) # 预测
            loss = loss_func(GT,preHSI) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 梯度更新
            loss_list.append(loss.item())
        test(epoch=epoch)
# 测试
@torch.no_grad()
def test(epoch=-1):
    model.eval()
    # 各项生成指标
    loss_list = []
    PSNR_list = []
    SSIM_list = []
    ERGAS_list = []
    SAM_list = []
    for idx,loader_data in enumerate(test_dataloader):
        GT,LRHSI,RGB = loader_data[0].to(device),loader_data[1].to(device),loader_data[2].to(device)
        preHSI = model(LRHSI,RGB) # 预测
        metric = Metric(GT,preHSI) # 计算图像生成指标
        loss = loss_func(GT,preHSI) # 损失函数
        loss_list.append(loss.item()) 
        PSNR_list.append(metric.PSNR)
        SSIM_list.append(metric.SSIM)
        ERGAS_list.append(metric.ERGAS)
        SAM_list.append(metric.SAM)
        
    if args.log_out == 1 and (epoch + 1) % args.check_step == 0: # 保存模型
        torch.save(model.state_dict(),f'{log_dir}/epoch:{epoch},PSNR:{np.array(PSNR_list).mean():.4f}.pth')
    # 各项指标取平均并且输出
    logger.info(f'[Test:{model_name},{args.dataset}] epoch:{epoch}, loss:{np.array(loss_list).mean():.4f}, PSNR:{np.array(PSNR_list).mean():.4f}, SSIM:{np.array(SSIM_list).mean():.4f}, SAM:{np.array(SAM_list).mean():.4f}, ERGAS:{np.array(ERGAS_list).mean():.4f}')

if __name__ == "__main__":
    train()