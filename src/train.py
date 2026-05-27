import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import build_model
from src.dataset import train_loader
from src.loss import FocalLoss
from src.validate import validate


def train_one_epoch(model,loader,loss_fn,optimizer,device):
    """
    Runs one full pass over the training set
    
    Steps per batch:
    1. zero_grad - clear accumulated gradients from previous batch
    2. forward - get model predictions
    3. loss - compute loss between predictions and true labels
    4. backward - compute gradients via backprop
    5. step - update weights using those grads
    
    return - avg_loss(float) - mean loss across all batches in this epoch
    """
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item() # .item pulls plain float from tensor

    avg_loss = running_loss/len(loader)
    return avg_loss
    
# exp 1 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    EPOCHS = 10
    LR = 1e-4
    NUM_CLASSES=15
    INPUT_DIM   = 20

    # ── Dummy loaders (same shape your real loaders will use) ─────────────────
    torch.manual_seed(42)
    X = torch.randn(400, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (400,))
    dummy_train = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    X_val = torch.randn(100, INPUT_DIM)
    y_val = torch.randint(0, NUM_CLASSES, (100,))
    dummy_val = DataLoader(TensorDataset(X_val, y_val), batch_size=32)


    model = build_model(num_classes=15).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    exp1_train_losses = []
    best_val_acc = 0.0

    for epoch in range(1,EPOCHS+1):
        train_loss = train_one_epoch(model,train_loader,loss_fn,optimizer,device)
        val_loss, val_acc = validate(model,val_loader,loss_fn, device)

        exp1_train_losses.append(round(train_loss,4))

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}%")
        
        #save best checkpoint
        if val_acc> best_val_acc:
            best_val_acc=val_acc
            torch.save(model.state_dict(),"results/checkpoints/checkpoint_ce.pth")
            print(f"✓ checkpoint_ce.pth saved (val_acc={val_acc*100:.2f}%)")


    print(f"\nexp1_train_losses = {exp1_train_losses}")
    print(f"Best Val Accuracy : {best_val_acc*100:.2f}%")

'''--------------------------------------------------------------------'''

print("\n── Experiment 2: FocalLoss(alpha=1, gamma=2) ──")
    model2     = build_model(num_classes=NUM_CLASSES).to(device)
    loss_fn2   = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=LR)

    exp2_train_losses = []
    exp2_val_losses   = []
    exp2_val_accs     = []

    for epoch in range(1, EPOCHS + 1):          # ← correct indentation
        train_loss       = train_one_epoch(model2, dummy_train, loss_fn2, optimizer2, device)
        val_loss, val_acc = validate(model2, dummy_val, loss_fn2, device)

        exp2_train_losses.append(round(train_loss, 4))
        exp2_val_losses.append(round(val_loss, 4))
        exp2_val_accs.append(round(val_acc, 4))

        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"\nexp2_train_losses = {exp2_train_losses}")
    print(f"exp2_val_losses   = {exp2_val_losses}")
    print(f"exp2_val_accs     = {exp2_val_accs}")
