import torch
import torch.nn as nn
import torchvision.models as models


""" 
Building a VGG16 based classifier for fruit recognition.
-Loads VGG16 pretrained
- Freeze the entire freature extractor(conv layers) 
- Replace only the final fully_connected layer with a new linear

"""

def build_model(num_classes=15):
    # Load pretrained VGG16
    # weights=VGG16_Weights.DEFAULT fetches the best available ImageNet weights
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Freeze all feature layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace final classifier layer
    # original was 1000 ImageNet classes but we took only 15 
    model.classifier[-1] = nn.Linear(4096, num_classes)

    return model

def count_parameters(model):
    """
    return a dict with total trainable and frozen param counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total,"trainable": trainable, "frozen": frozen}

if __name__ == "__main__":
    model = build_model()
    counts = count_parameters(model)

    print(f"Total params: {counts['total']:,}")
    print(f"Frozen params: {counts['frozen']:,}")
    print(f"Trainable params: {counts['trainable']:,}")

    #forward pass on a dummy image
    dummy = torch.randn(1,3,224,224)
    output = model(dummy)
    assert output.shape == (1,15), f"Expected (1,15), got {output.shape}"
    print(f"Output shape:  {output.shape}")