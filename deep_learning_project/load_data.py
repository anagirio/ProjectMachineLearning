import fix_images
import numpy as np
import torch
import torchvision
import argparse
import time
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import multiprocessing
import sys
from net import Net

train_dir = './train_images'
test_dir = './test_images'

# TRANSFORMATIONS WITH DATA AUGMENTATION FOR TRAINING
# These transformations artificially increase the variability of the data
transform_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomRotation(15),              # Random rotation ±15 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.RandomHorizontalFlip(p=0.5),     # Horizontal flip 50% of the time
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Brightness/contrast jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=(0,), std=(1,)),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))  # Simulate occlusions (objects in front)
])

# Transformation for validation/test (WITHOUT augmentation - original data)
transform_val = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0,), std=(1,))
])

# Training dataset with augmentation
train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
# Validation dataset WITHOUT augmentation (for proper evaluation)
train_data_val = torchvision.datasets.ImageFolder(train_dir, transform=transform_val)
test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform_val)

valid_size = 0.2
batch_size = 32

num_train = len(train_data)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

train_sampler = SubsetRandomSampler(train_new_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

classes = ('noface','face')


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # output: (N,1) -> make it (N,) to match target shape for BCELoss
            output_flat = output.view(-1)
            # BCELoss expects float targets
            loss = criterion(output_flat, target.float())
            running_loss += loss.item()
            n_batches += 1
            
            # Calculate accuracy
            # Prediction by threshold (0.5)
            predicted = (output.data >= 0.5).long().view(-1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = running_loss / n_batches if n_batches > 0 else float('inf')
    accuracy = 100 * correct / total if total > 0 else 0
    model.train()
    return avg_loss, accuracy


def build_dataloaders(batch_size=batch_size, num_workers=None):
    """Return (train_loader, valid_loader, test_loader).

    Train loader uses data augmentation; valid/test do not.
    """
    if num_workers is None:
        num_workers = 0 if sys.platform.startswith('win') else 1

    # Train with augmentation
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=num_workers
    )
    
    # Validation WITHOUT augmentation
    valid_loader = torch.utils.data.DataLoader(
        train_data_val, 
        batch_size=batch_size, 
        sampler=valid_sampler, 
        num_workers=num_workers
    )
    
    # Test WITHOUT augmentation
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    # images, labels = next(iter(train_loader))
    # print(f"Image shape: {images.shape}")

    return train_loader, valid_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Training with data augmentation and improved hyperparameters')
    parser.add_argument('--batch-size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=30 , help='More epochs for better learning with augmentation')
    parser.add_argument('--lr', type=float, default=0.001, help='Lower learning rate for stability')
    parser.add_argument('--num-workers', type=int, default=(0 if sys.platform.startswith('win') else 1))
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--save-path', type=str, default='best_model_augmented.pth')
    args = parser.parse_args()

    device = torch.device('cpu')

    # Create DataLoaders
    train_loader, valid_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    # Initialize model
    net = Net().to(device)
    criterion = nn.BCELoss()
    
    # Optimizer with momentum and weight decay for better generalization
    optimizer = optim.SGD(
        net.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    
    # Learning rate scheduler - reduces LR when validation stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    n_epochs = args.epochs
    best_val = float('inf')
    epochs_no_improve = 0

    print(f'Starting training with data augmentation')
    print(f'Training samples: {len(train_new_idx)}')
    print(f'Validation samples: {len(valid_idx)}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Max epochs: {n_epochs}')
    print('-' * 60)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        net.train()
        running_loss = 0.0
        i = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            # Flatten output to (N,) and convert target to float for BCELoss
            output_flat = output.view(-1)
            loss = criterion(output_flat, target.float())
            loss.backward()
            optimizer.step()
            i += 1
            running_loss += loss.item()
            
            if i % 200 == 199:
                print(f'[{epoch}, {i}] train loss (recent): {running_loss / 200:.6f}')
                running_loss = 0.0

        # Evaluate on validation set
        val_loss, val_acc = evaluate(net, valid_loader, criterion, device)
        
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)
        
        elapsed = time.time() - t0
        print(f'Epoch {epoch}/{n_epochs} finished in {elapsed:.1f}s')
        print(f'  Validation loss: {val_loss:.6f}')
        print(f'  Validation accuracy: {val_acc:.2f}%')

        # Early stopping and checkpoint
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(net.state_dict(), args.save_path)
            print(f'  ✓ Validation improved! Saved model to {args.save_path}')
        else:
            epochs_no_improve += 1
            print(f'  No improvement for {epochs_no_improve} epoch(s)')

        if epochs_no_improve >= args.patience:
            print(f'\nEarly stopping triggered (no improvement in {args.patience} epochs).')
            break
        
        print('-' * 60)

    print('\n' + '=' * 60)
    print('Finished Training!')
    print(f'Best validation loss: {best_val:.6f}')
    print(f'Model saved to: {args.save_path}')
    print('=' * 60)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

