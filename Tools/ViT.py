## from https://github.com/lucidrains/vit-pytorch  主要用于性能预测网络
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torch import nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import math
class PositionalEncoding(nn.Module):
    """
    位置编码模块
    输入:
    - embed_dim: 嵌入维度
    - max_len: 最大长度
    输出:
    - 输出张量
    """
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        # 创建一个大的固定位置编码表
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 调整形状以匹配输入特征形状 (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x的形状为 (batch_size, seq_len, embed_dim)
        # 需要将 pe 的形状从 (1, max_len, embed_dim) 切片并扩展为 (batch_size, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1), :]
        return x

class PreNorm(nn.Module):
    """预标准化
    输入:(batch_size, num_patches, dim)
    输出:(batch_size, num_patches, dim)
    - dim: 输入维度
    - fn: 模块
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)#层标准化
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """
前馈神经网络
    输入:(batch_size, num_patches, dim)
    输出:(batch_size, num_patches, dim)
    - dim: 输入维度
    - hidden_dim: 中间层维度
    - dropout: 丢弃率
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):#heads头数 dim_head每个头维度
    def __init__(self, dim, heads = 16, dim_head = 64, dropout = 0.1):
        """
        计算多头自注意力
        输入:(batch_size, num_patches, dim)
        输出:(batch_size, num_patches, dim)
        - dim: 输入每个patch长度
        - heads: 头数
        - dim_head: 每个头的维度
        - dropout: 丢弃率 防止过拟合
        :param dim: 输入每个patch长度
        :param heads: 头数
        :param dim_head: 每个头的维度
        :param dropout: 丢弃率 防止过拟合
        """
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)#检查是否需要投影

        self.heads = heads
        self.scale = dim_head ** -0.5#计算缩放因子

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x): ## 最重要的都是forword函数了
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        ## 对tensor张量分块 x :1 197 1024   qkv 最后 是一个元组，tuple，长度是3，每个元素形状：1 197 1024
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # 分成多少个Head,与TRM生成qkv 的方式不同， 要更简单，不需要区分来自Encoder还是Decoder
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale#计算点积
        attn = self.attend(dots)#计算注意力
        out = torch.matmul(attn, v)#计算输出
        out = rearrange(out, 'b h n d -> b n (h d)') #合并多个头
        return self.to_out(out)

class Transformer(nn.Module):
    """
    多头自注意力
    输入:(batch_size, num_patches, dim)
    输出:(batch_size, num_patches, dim)
    -dim: 输入每个patch长度
    -depth: 残差块数量
    -heads: 头数
    -dim_head: 每个头的维度
    -mlp_dim: mlp层维度
    -dropout: 丢弃率
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),#层归一化后求注意力
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))#层归一化后前馈
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
# 1. VIＴ整体架构从这里开始
class ViT(nn.Module):#dim是每个补丁的维度
    def __init__(self, *,lenth,num_classes, dim, depth, heads, mlp_dim, pos='learn',pool = 'mean', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        """
        输入：(batch_size,  channels, lenth)
        输出：(batch_size,  num_classes)
        :param lenth:
        :param num_classes:
        :param dim: 将lenth-->dim
        :param depth: 深度
        :param heads: 头数
        :param pos: 位置编码类型，默认可学习
        :param mlp_dim: 前馈网络中间层长度
        :param pool: 池化类型
        :param channels: 输入通道数即几个patch
        :param dim_head: 每头维度
        :param dropout:
        :param emb_dropout:
        """
        super().__init__()
        self.pos=pos
        if pos == 'sin':
            self.pos_embedding1 = PositionalEncoding(dim, 30)
        elif pos =='learn':
            self.pos_embedding = nn.Parameter(torch.randn(1, channels + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)  ,
        )
        self.liear=nn.Linear(lenth, dim)
    def forward(self, x):
        x=self.liear(x)
        b, n, _ = x.shape ##
        #将cls 复制 batch_size 份
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # 将cls token在维度1 扩展到输入上
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置编码
        if self.pos == 'learn':
            x += self.pos_embedding[:, :(n + 1)]
        elif self.pos == 'sin':
            x=self.pos_embedding1(x)
        x = self.dropout(x)
        # 输入TRM
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] #求均值或者最后一个
        x = self.to_latent(x)
        return self.mlp_head(x)

if __name__ == '__main__':
    '''image_size: 输入图像的尺寸。
    patch_size: 图像被分为的小块的尺寸。
    num_classes: 输出的类别数量。
    dim: 补丁的维度。
    depth: Transformer的层数。
    heads: 注意力头的数量。
    mlp_dim: 前馈神经网络中间隐藏层的维度。
    pool: 池化的类型，可以是
    'cls'（使用类别标记进行池化）或
    'mean'（使用平均池化）。
    channels: 输入图像的通道数。
    dim_head: 每个头的维度。
    dropout: Dropout的比例。
    emb_dropout: 嵌入层的Dropout比例。'''
    model = ViT(
        lenth =3,
        channels=7,
        num_classes =2,
        dim =512,
        depth =4,
        heads = 9,
        mlp_dim = 1024,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to("cuda")
    summary(model, input_size=(7, 3),device="cuda")
