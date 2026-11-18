import torch
from cardionet.models import ECGCNN, ECGResNet, build_model

def test_cnn_output_shape():
    model = ECGCNN()
    assert model(torch.randn(2, 12, 1000)).shape == (2, 5)

def test_resnet_output_shape():
    model = ECGResNet()
    assert model(torch.randn(2, 12, 1000)).shape == (2, 5)

def test_factory_rejects_unknown():
    try:
        build_model("transformer")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown model must raise ValueError")
