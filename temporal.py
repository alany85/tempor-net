import math
from datetime import datetime
import torch

def encode_timestamp(timestamp_ms: int) -> torch.Tensor:
    # ms to s to datetime
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
    month = dt.month  # 1-12
    hour = dt.hour    # 0-23
    
    # cyclic encoding
    month_sin = math.sin(2 * math.pi * month / 12.0)
    month_cos = math.cos(2 * math.pi * month / 12.0)
    
    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)
    
    return torch.tensor([month_sin, month_cos, hour_sin, hour_cos], dtype=torch.float32)