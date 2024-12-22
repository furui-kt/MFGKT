"""
数据设置
"""
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOG_B = '\033[1;34m' # 蓝色
LOG_Y = '\033[1;33m' # 黄色
LOG_G = '\033[1;36m' # 深绿色
LOG_END = '\033[m' # 结束标记

size_q_feature = 8
size_embedding =200 