import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def get_model(num_classes):
    return fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=num_classes)