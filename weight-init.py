import math
import torch.nn as nn

def w_init_heuristic(fan_in, fan_out):
    return math.sqrt(1 / (3 * fan_in))

def w_init_xavier(fan_in, fan_out):
    return math.sqrt(2 / (fan_in + fan_out))

def w_init_xavier_caffe(fan_in, fan_out):
    return math.sqrt(1 / fan_in)

def w_init_kaiming(fan_in, fan_out):
    return math.sqrt(4 / (fan_in + fan_out))

def w_init(net, method_name):
    # 选择初始化方法
    if method_name == 'heuristic':
        method = w_init_heuristic
    elif method_name == 'xavier':
        method = w_init_xavier
    elif method_name == 'xavier_caffe':
        method = w_init_xavier_caffe
    elif method_name == 'kaiming':
        method = w_init_kaiming
    else:
        raise ValueError("Unknown initialization method")

    # 遍历所有卷积模块
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Conv1d)):
            fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            fan_out = m.out_channels * m.kernel_size[0] * m.kernel_size[1]
            m.weight.data.normal_(0, method(fan_in, fan_out))
        elif isinstance(m, nn.Linear):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            m.weight.data.normal_(0, method(fan_in, fan_out))
        if m.bias is not None:
            m.bias.data.zero_()

    return net