import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "models", "tumor_classifier.pth")
BATCH_SIZE = 32
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 20
LR_PHASE1 = 0.001
LR_PHASE2 = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    print(f"[CONFIG] Using device: {DEVICE}")
    print(f"[CONFIG] Data directory: {DATA_DIR}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"[DATA] Classes found: {train_dataset.classes}")
    print(f"[DATA] Class mapping: {train_dataset.class_to_idx}")
    print(f"[DATA] Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    for cls_name in train_dataset.classes:
        cls_idx = train_dataset.class_to_idx[cls_name]
        count = sum(1 for _, y in train_dataset.samples if y == cls_idx)
        print(f"[DATA]   Train {cls_name}: {count} images")

    dataloaders = {'train': train_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # ---------- PHASE 1: Train only FC layer (freeze backbone) ----------
    print("\n" + "="*60)
    print("[PHASE 1] Training classifier head only - freezing backbone")
    print("="*60)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer_p1 = optim.Adam(model.fc.parameters(), lr=LR_PHASE1, weight_decay=1e-4)
    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1, mode='max', factor=0.5, patience=2)

    for epoch in range(PHASE1_EPOCHS):
        print(f'\nPhase 1 - Epoch {epoch+1}/{PHASE1_EPOCHS}')
        print('-' * 40)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer_p1.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer_p1.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if i % 15 == 0 and phase == 'train':
                    pct = 100 * running_corrects.double() / ((i + 1) * BATCH_SIZE)
                    print(f'  Batch {i:3d}/{len(dataloaders[phase])} | Loss: {loss.item():.4f} | Acc: {pct:.2f}%')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'  >> {phase.upper()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%')

            if phase == 'test':
                scheduler_p1.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f'  >> NEW BEST accuracy: {best_acc*100:.2f}%')

    # ---------- PHASE 2: Fine-tune entire network ----------
    print("\n" + "="*60)
    print("[PHASE 2] Fine-tuning full network - unfreezing all layers")
    print("="*60)
    for param in model.parameters():
        param.requires_grad = True

    optimizer_p2 = optim.Adam(model.parameters(), lr=LR_PHASE2, weight_decay=1e-4)
    scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, mode='max', factor=0.5, patience=3)

    for epoch in range(PHASE2_EPOCHS):
        print(f'\nPhase 2 - Epoch {epoch+1}/{PHASE2_EPOCHS}')
        print('-' * 40)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer_p2.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer_p2.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if i % 15 == 0 and phase == 'train':
                    pct = 100 * running_corrects.double() / ((i + 1) * BATCH_SIZE)
                    print(f'  Batch {i:3d}/{len(dataloaders[phase])} | Loss: {loss.item():.4f} | Acc: {pct:.2f}%')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'  >> {phase.upper()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%')

            if phase == 'test':
                scheduler_p2.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f'  >> NEW BEST accuracy: {best_acc*100:.2f}%')

    # ---------- Save best model ----------
    print("\n" + "="*60)
    model.load_state_dict(best_model_wts)
    if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH))
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[DONE] Best model saved to: {MODEL_SAVE_PATH}")
    print(f"[DONE] Best test accuracy: {best_acc*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    train_model()
