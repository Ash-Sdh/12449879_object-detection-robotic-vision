import torch
from models.cbam import CBAM

def test_cbam_shape_preserved():
    x = torch.randn(2, 64, 32, 32)
    m = CBAM(64)
    y = m(x)
    assert y.shape == x.shape
