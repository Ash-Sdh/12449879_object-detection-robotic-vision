import torch

def test_shapes():
    x = torch.zeros(1, 3, 640, 640)
    assert x.shape == (1, 3, 640, 640)
    assert x.dtype == torch.float32
