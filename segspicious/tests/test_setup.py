"""Smoke tests verifying the environment is correctly set up."""


def test_numpy_import():
    import numpy as np

    a = np.array([1, 2, 3])
    assert a.sum() == 6


def test_torch_import():
    import torch

    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.sum().item() == 6.0


def test_torchvision_import():
    import torchvision

    assert hasattr(torchvision, "datasets")


def test_segspicious_import():
    import segspicious

    assert "protocols" in segspicious.__doc__.lower() or "segspicious" in segspicious.__doc__.lower()
