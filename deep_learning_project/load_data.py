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

transform = transforms.Compose(
    [transforms.Grayscale(), 
     transforms.ToTensor(), 
     transforms.Normalize(mean=(0,),std=(1,))])

train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

valid_size = 0.2
batch_size = 32

num_train = len(train_data)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

train_sampler = SubsetRandomSampler(train_new_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# On Windows the 'spawn' start method is used by default which requires
# that the main module be import-safe. We'll create DataLoaders inside main()
# and default to 0 workers on Windows to avoid bootstrapping errors.
classes = ('noface','face')


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            n_batches += 1
    avg_loss = running_loss / n_batches if n_batches > 0 else float('inf')
    model.train()
    return avg_loss

def build_dataloaders(batch_size=batch_size, num_workers=None):
    """Return (train_loader, valid_loader, test_loader).

    This function is safe to import from other modules (it doesn't run
    training) and can be used by `test.py` to obtain the test loader.
    """
    if num_workers is None:
        num_workers = 0 if sys.platform.startswith('win') else 1

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, valid_loader, test_loader


def main():
    # aqui é para passagem de argumento, maioria já tem default ai n precisa passar nd, botei 2 epocas pq tinha visto num dos tutoriais q uso isso, mas da pra brincar
    parser = argparse.ArgumentParser(description='Training with validation and early stopping')
    parser.add_argument('--batch-size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num-workers', type=int, default=(0 if sys.platform.startswith('win') else 1))
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience in epochs')
    parser.add_argument('--save-path', type=str, default='best_model.pth')
    args = parser.parse_args()

    # aq frescurinha pra roda com gpu pode ate apagar
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # Create DataLoaders inside main so worker spawning happens safely
    train_loader, valid_loader, test_loader = build_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)

    # aq começa realemnte o q deve se feito
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    n_epochs = args.epochs

    best_val = float('inf')
    epochs_no_improve = 0

    #esse é o for pra ser preenchido

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        net.train()
        running_loss = 0.0
        i = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            i += 1
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'[{epoch}, {i}] train loss (recent): {running_loss / 200:.6f}')
                running_loss = 0.0

        val_loss = evaluate(net, valid_loader, criterion, device)
        print(f'Epoch {epoch} finished in {time.time()-t0:.1f}s - validation loss: {val_loss:.6f}')

        # early stopping / checkpoint <- logica de parar no melhor valor
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(net.state_dict(), args.save_path)
            print(f'Validation improved; saved model to {args.save_path}')
        else:
            epochs_no_improve += 1
            print(f'No improvement for {epochs_no_improve} epoch(s)')

        if epochs_no_improve >= args.patience:
            print(f'Early stopping triggered (no improvement in {args.patience} epochs).')
            break

    print('Finished Training')


if __name__ == '__main__':
    # Required on Windows when using multiprocessing in frozen/executable apps
    #tava dando erro de freeze nessa bosta, ai tive q por isso ae
    multiprocessing.freeze_support()
    main()


