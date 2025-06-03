import torch
import torch.nn as nn
import timm

class CrowdCounter(nn.Module):
    def __init__(self, block_size=4, authentic_weights_path=None, synthetic_weights_path=None):
        """
        模型初始化
        Args:
            block_size: 模型块大小
            authentic_weights_path: 真实分支预训练权重路径
            synthetic_weights_path: 合成分支预训练权重路径
        """
        super(CrowdCounter, self).__init__()

        # 双分支架构
        self.authentic = timm.create_model("mobilenetv3_large_100.ra_in1k", pretrained=False)
        self.syntetic = timm.create_model("mobilenetv3_large_100.ra_in1k", pretrained=False)
        
        if authentic_weights_path:
            self.authentic.load_state_dict(torch.load(authentic_weights_path, map_location=torch.device('cpu')))
        
        if synthetic_weights_path:
            self.syntetic.load_state_dict(torch.load(synthetic_weights_path, map_location=torch.device('cpu')))
        
        # 调整网络架构
        self.aut_up = nn.Linear(1000, 3100)
        self.syn_up = nn.Linear(1000, 3100)
        self.aut_dw = nn.Linear(3100, 512)
        self.syn_dw = nn.Linear(3100, 512)
        
        # 输出层，用于整数预测，移除了sigmoid激活
        self.head = nn.Linear(1024, 1)
        self.relu = nn.ReLU()

    def forward(self, inp, inp2):
        """
        前向传播
        Args:
            inp: 输入图像（真实分支）
            inp2: 输入图像（合成分支）
        Returns:
            输出人数预测值
        """
        authentic = self.authentic(inp)
        authentic = self.relu(authentic) 
        authentic = self.aut_up(authentic)
        authentic = self.relu(authentic)
        authentic = self.aut_dw(authentic)
        authentic = self.relu(authentic)

        syntetic = self.syntetic(inp2)
        syntetic = self.relu(syntetic)
        syntetic = self.syn_up(syntetic)
        syntetic = self.relu(syntetic)
        syntetic = self.syn_dw(syntetic)
        syntetic = self.relu(syntetic)

        concat_pool = torch.cat([authentic, syntetic], dim=1)
        output = self.head(concat_pool)

        return output 