import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# 定义时间嵌入层
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(1, embed_dim)  # 将时间映射到高维空间
        self.linear2 = nn.Linear(embed_dim, embed_dim)  # 进一步映射
        self.activation = nn.SiLU()  # 激活函数

    def forward(self, t):
        # 输入: t (batch_size, 1)
        # 输出: (batch_size, embed_dim)
        t = self.linear1(t)  # (batch_size, embed_dim)
        t = self.activation(t)
        t = self.linear2(t)  # (batch_size, embed_dim)
        return t

# 定义条件参数处理的多层感知机 (MLP)
class ConditionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ConditionMLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)  # 第一层线性变换
        self.linear2 = nn.Linear(hidden_dim, output_dim)  # 第二层线性变换
        self.activation = nn.SiLU()  # 激活函数

    def forward(self, x):
        # 输入: x (batch_size, input_dim)
        # 输出: (batch_size, output_dim)
        x = self.linear1(x)  # (batch_size, hidden_dim)
        x = self.activation(x)
        x = self.linear2(x)  # (batch_size, output_dim)
        return x

# 定义一维卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)  # 一维卷积
        self.norm = nn.BatchNorm1d(out_channels)  # 批归一化
        self.activation = nn.SiLU()  # 激活函数

    def forward(self, x):
        # 输入: x (batch_size, in_channels, length)
        # 输出: (batch_size, out_channels, length)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

# 定义下采样块
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)  # 卷积块
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 最大池化
        self.time_embed_proj = nn.Linear(time_embed_dim, out_channels)  # 时间嵌入投影

    def forward(self, x, t):
        # 输入: x (batch_size, in_channels, length), t (batch_size, time_embed_dim)
        # 输出: x (batch_size, out_channels, length // 2), skip (batch_size, out_channels, length)
        x = self.conv(x)
        skip = x  # 保存跨连接
        x = self.pool(x)
        # 引入时间嵌入
        t = self.time_embed_proj(t)  # (batch_size, out_channels)
        t = t.unsqueeze(-1)  # (batch_size, out_channels, 1)
        x = x + t  # 将时间嵌入加到特征上
        return x, skip

# 定义上采样块
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)  # 转置卷积
        self.conv = ConvBlock(out_channels * 2, out_channels)  # 卷积块
        self.time_embed_proj = nn.Linear(time_embed_dim, out_channels)  # 时间嵌入投影

    def forward(self, x, skip, t):
        # 输入: x (batch_size, in_channels, length), skip (batch_size, out_channels, length * 2), t (batch_size, time_embed_dim)
        # 输出: (batch_size, out_channels, length * 2)
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # 跨连接
        x = self.conv(x)

        # 引入时间嵌入
        t = self.time_embed_proj(t)  # (batch_size, out_channels)
        t = t.unsqueeze(-1)  # (batch_size, out_channels, 1)
        x = x + t  # 将时间嵌入加到特征上

        return x

# 定义一维U-Net网络
class UNet1D(nn.Module):
    def __init__(self, input_dim, time_embed_dim=256, condition_dim=128):
        super(UNet1D, self).__init__()
        # 输入处理
        self.input_dim=input_dim
        self.input_linear = nn.Linear(input_dim, 256)  # 线性层将输入映射到256维
        self.input_conv = ConvBlock(1, 64)  # 卷积层将1通道变为64通道

        # 下采样路径
        self.down1 = DownBlock(64, 128, time_embed_dim)  # 下采样块1
        self.down2 = DownBlock(128, 256, time_embed_dim)  # 下采样块2
        self.down3 = DownBlock(256, 512, time_embed_dim)  # 下采样块3

        # 中间层
        self.mid_conv = ConvBlock(512, 1024)  # 中间卷积块

        # 上采样路径
        self.up1 = UpBlock(1024, 512, time_embed_dim)  # 上采样块1
        self.up2 = UpBlock(512, 256, time_embed_dim)  # 上采样块2
        self.up3 = UpBlock(256, 128, time_embed_dim)  # 上采样块3

        # 输出层
        self.output_conv = nn.Conv1d(128, 1, kernel_size=1)  # 输出卷积层

        # 时间嵌入
        self.time_embed = TimeEmbedding(time_embed_dim)  # 时间嵌入层

        # 条件参数处理
        self.condition_mlp = ConditionMLP(condition_dim, 256, time_embed_dim)  # 条件参数MLP
        self.out=nn.Linear(256,21)

    def forward(self, x, t, condition):
        # 输入: x (batch_size, input_dim), t (batch_size, 1), condition (batch_size, condition_dim)
        # 输出: (batch_size, input_dim)
        x=x.reshape(-1,self.input_dim)
        t=t.reshape(-1,1)
        t = t.to(dtype=torch.float32)
        # print(x.shape,t.shape,condition.shape)
        # 输入处理
        x = self.input_linear(x)  # (batch_size, 256)
        x = x.unsqueeze(1)  # (batch_size, 1, 256)
        x = self.input_conv(x)  # (batch_size, 64, 256)

        # 时间嵌入
        t = self.time_embed(t)  # (batch_size, time_embed_dim)

        # 条件参数处理
        condition = self.condition_mlp(condition)  # (batch_size, time_embed_dim)

        # 将条件参数加到时间嵌入中
        t = t + condition  # (batch_size, time_embed_dim)

        # 下采样
        x, skip1 = self.down1(x, t)  # (batch_size, 128, 128)
        x, skip2 = self.down2(x, t)  # (batch_size, 256, 64)
        x, skip3 = self.down3(x, t)  # (batch_size, 512, 32)

        # 中间层
        x = self.mid_conv(x)  # (batch_size, 1024, 32)

        # 上采样
        x = self.up1(x, skip3, t)  # (batch_size, 512, 64)
        x = self.up2(x, skip2, t)  # (batch_size, 256, 128)
        x = self.up3(x, skip1, t)  # (batch_size, 128, 256)

        # 输出
        x = self.output_conv(x)  # (batch_size, 1, 256)
        x = x.squeeze(1)  # (batch_size, 256)
        x=self.out(x).unsqueeze(2)
        return x

# 测试网络
if __name__ == "__main__":
    input_dim = 208
    time_embed_dim = 256
    condition_dim = 1
    batch_size = 8

    model = UNet1D(input_dim, time_embed_dim, condition_dim)
    x = torch.randn(batch_size, input_dim)  # 输入
    t = torch.randn(batch_size,1)  # 时间
    condition = torch.randn(batch_size, condition_dim)  # 条件参数

    # 使用 torchsummary 查看模型结构
    summary(model, [(input_dim,), (1,), (condition_dim,)], device="cpu")

    # 前向传播
    output = model(x, t, condition)
    print("Output shape:", output.shape)  # 输出形状: (batch_size, input_dim)