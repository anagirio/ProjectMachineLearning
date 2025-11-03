import fix_images
import torch
from net import Net
from load_data import build_dataloaders
import os

def test(model_path='best_model_augmented.pth', batch_size=32, num_workers=0):
    """
    Testa o modelo treinado com data augmentation.
    
    Args:
        model_path: caminho do modelo (best_model_augmented.pth)
        batch_size: tamanho do batch
        num_workers: número de workers para dataloader
    """
    print(f'Loading test data...')
    train_loader, valid_loader, test_loader = build_dataloaders(batch_size=batch_size, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    net = Net().to(device)
    
    if model_path and os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Loaded model from {model_path}')
    else:
        print(f'ERROR: Model file not found at {model_path}')
        print('Please train the model first using:')
        print('  python load_data_augmented.py --epochs 30')
        return
    
    net.eval()

    correct = 0
    total = 0
    
    # Contadores por classe
    class_correct = [0, 0]  # [noface, face]
    class_total = [0, 0]
    
    print('\nTesting model...')
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Acurácia por classe
            for label, prediction in zip(labels, predicted):
                class_total[label] += 1
                if label == prediction:
                    class_correct[label] += 1

    accuracy = 100 * correct / total if total > 0 else 0
    
    print('\n' + '='*60)
    print('Test Results:')
    print('='*60)
    print(f'Overall Accuracy: {accuracy:.2f}% ({correct}/{total})')
    print('-'*60)
    
    classes = ['noface (0)', 'face (1)']
    for i in range(2):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f'Accuracy for {classes[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')
        else:
            print(f'Accuracy for {classes[i]}: N/A (no samples)')
    
    print('='*60)


if __name__ == '__main__':
    test(model_path='best_model_augmented.pth', batch_size=32, num_workers=0)
