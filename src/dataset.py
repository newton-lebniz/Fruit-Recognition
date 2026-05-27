from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#This is for image transformations..
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#Loads datasets separately..
train_dataset = datasets.ImageFolder(
    root="data/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root="data/val",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="data/test",
    transform=transform
)

#this variable creates dataloaders..
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

#Test one batch
images, labels = next(iter(train_loader))

print("Image batch shape:")
print(images.shape)