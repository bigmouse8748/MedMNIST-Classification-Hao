import torch.nn as nn
import torchvision.models as models

def get_stock_model(model_name, num_classes, pretrained=True):
    """
    Returns a stock model from torchvision with a customizable backbone.

    Args:
        model_name (str): Name of the backbone model (e.g., 'resnet50', 'densenet121', 'efficientnet_b0').
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        torch.nn.Module: A model with the specified backbone and adjusted output layer.
    """
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=pretrained)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Supported options are: "
                         f"'resnet50', 'resnet18', 'efficientnet_b0', 'mobilenet_v3_large'.")

    return model