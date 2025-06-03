from .mobile_kan_crowd_counter import (
    MobileKANCrowdCounter,
    create_mobile_kan_crowd_counter,
    CompositionLoss
)

__all__ = [
    'MobileKANCrowdCounter',
    'create_mobile_kan_crowd_counter',
    'CompositionLoss'
]

def create_mobile_kan_crowd_counter(config=None):
    """创建轻量级MobileKAN人群计数模型"""
    # 默认配置
    default_config = {
        'backbone': 'mobilenetv3_large_100',
        'pretrained': True,
        'pretrained_path': None,
        'dropout': 0.0
    }
    
    # 使用提供的配置覆盖默认配置
    if config is not None:
        default_config.update(config)
    
    model = MobileKANCrowdCounter(
        backbone=default_config['backbone'],
        pretrained=default_config['pretrained'],
        pretrained_path=default_config['pretrained_path'],
        dropout=default_config['dropout']
    )
    
    return model 