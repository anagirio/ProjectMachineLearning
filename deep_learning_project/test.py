import torch
from net import Net
from load_data import build_dataloaders
import os

def test(model_path=None, batch_size=32, num_workers=0):
    train_loader, valid_loader, test_loader = build_dataloaders(batch_size=batch_size, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)
    if model_path and os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Loaded model from {model_path}')
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total if total>0 else 0))


if __name__ == '__main__':
    # adjust model_path if you saved to a different file name
    test(model_path='best_model.pth', batch_size=32, num_workers=0)

