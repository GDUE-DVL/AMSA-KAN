# models 模块

__version__ = "0.1.0"

from .iqa.lar_iqa_model import LARIQACrowdCounter, LARIQANetwork
from .crowd_counter_with_kan import CrowdCounterWithKAN, DualPathCrowdCounter, MobileNetMergedCrowdCounter

try:
    from .advanced_crowd_counter import AdvancedCrowdCounterWithKAN, DualStreamCrowdCounter
except ImportError:
    pass

__all__ = [
    "LARIQACrowdCounter",
    "LARIQANetwork", 
    "CrowdCounterWithKAN",
    "DualPathCrowdCounter",
    "MobileNetMergedCrowdCounter",
    "AdvancedCrowdCounterWithKAN",
    "DualStreamCrowdCounter",
]
