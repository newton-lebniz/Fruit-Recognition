import torch
import torch.nn as nn
import torchvision.models as models

def build_model(num_classes=15):
    # Load pretrained VGG16
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Freeze all feature layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace final classifier layer
    model.classifier[-1] = nn.Linear(4096, num_classes)

    return model

if __name__ == "__main__":
    model = build_model()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(
        p.numel() for p in model.parameters()
        if not p.requires_grad
    )
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total params:     {total_params:,}")
    print(f"Frozen params:    {frozen_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Test with dummy input
    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 15), "Shape mismatch!"
    print("All checks passed.")