import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        
    def forward(self, x1, x2):
        # 确保x1和x2尺寸一致
        if x1.size() != x2.size():
            x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.se(x)
        return x

class MobileNetMergedAttention(nn.Module):
    def __init__(self, authentic_weights_path=None, synthetic_weights_path=None):
        super(MobileNetMergedAttention, self).__init__()
        
        # 使用更强大的骨干网络
        self.authentic = timm.create_model("efficientnet_b0", pretrained=True, features_only=True)
        self.synthetic = timm.create_model("efficientnet_b0", pretrained=True, features_only=True)
        
        if authentic_weights_path:
            self.authentic.load_state_dict(torch.load(authentic_weights_path, map_location=torch.device('cpu')))
        
        if synthetic_weights_path:
            self.synthetic.load_state_dict(torch.load(synthetic_weights_path, map_location=torch.device('cpu')))
        
        # 获取特征提取器的输出维度
        self.feature_dims = [32, 24, 40, 112, 320]  # EfficientNet-B0的特征维度
        
        # 添加通道注意力
        self.ca_auth = ChannelAttention(self.feature_dims[-1])
        self.ca_synth = ChannelAttention(self.feature_dims[-1])
        
        # 添加空间注意力
        self.sa_auth = SpatialAttention()
        self.sa_synth = SpatialAttention()
        
        # 特征融合层
        self.fusion = FeatureFusion(self.feature_dims[-1] * 2, self.feature_dims[-1])
        
        # 全局池化后的特征转换
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 特征转换层
        self.fc_auth = nn.Sequential(
            nn.Linear(self.feature_dims[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.fc_synth = nn.Sequential(
            nn.Linear(self.feature_dims[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 头部
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x, model):
        features = model(x)
        return features[-1]  # 取最后一层特征
    
    def forward(self, auth_input, synth_input):
        # 特征提取
        auth_feat = self.extract_features(auth_input, self.authentic)
        synth_feat = self.extract_features(synth_input, self.synthetic)
        
        # 应用通道注意力
        auth_ca = self.ca_auth(auth_feat) * auth_feat
        synth_ca = self.ca_synth(synth_feat) * synth_feat
        
        # 应用空间注意力
        auth_sa = self.sa_auth(auth_ca) * auth_ca
        synth_sa = self.sa_synth(synth_ca) * synth_ca
        
        # 全局池化
        auth_pool = self.avg_pool(auth_sa).view(auth_sa.size(0), -1)
        synth_pool = self.avg_pool(synth_sa).view(synth_sa.size(0), -1)
        
        # 特征转换
        auth_fc = self.fc_auth(auth_pool)
        synth_fc = self.fc_synth(synth_pool)
        
        # 特征融合
        concat_features = torch.cat([auth_fc, synth_fc], dim=1)
        
        # 最终输出
        output = self.head(concat_features)
        
        return output 