import torch
import torch.optim as optim
import torch.nn as nn
import math
from dataset import CIFAR10Dataset, AlbumentationsTransform
from model import CIFAR10Model
from utils import TrainingLogger
from torch.optim.lr_scheduler import LambdaLR

def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))) * (1 - 1e-6) + 1e-6
    
    return LambdaLR(optimizer, lr_lambda)

def train_model():
    # CIFAR-10 mean and std values
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    print("Preparing datasets...")
    transform = AlbumentationsTransform(mean=mean, std=std)
    
    trainset = CIFAR10Dataset(
        root='./data', 
        train=True,
        transform=transform,
        download=True
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=128,
        shuffle=True, 
        num_workers=2
    )

    testset = CIFAR10Dataset(
        root='./data', 
        train=False,
        transform=transform,
        download=True
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=128,
        shuffle=False, 
        num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CIFAR10Model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=5e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    
    num_epochs = 50
    warmup_epochs = 10
    scheduler = get_lr_scheduler(optimizer, warmup_epochs, num_epochs)
    
    logger = TrainingLogger()
    print("Starting training...")
    
    try:
        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(trainloader, 1):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if i % 50 == 0:
                    train_acc = 100 * correct / total
                    avg_loss = running_loss / i
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.log_step(epoch, i, train_acc, avg_loss, current_lr, len(trainloader))
            
            # Evaluate on test set
            model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            test_acc = 100 * test_correct / test_total
            avg_loss = running_loss / len(trainloader)
            current_lr = optimizer.param_groups[0]['lr']
            
            logger.log_epoch(epoch, train_acc, test_acc, avg_loss, current_lr, len(trainloader))
            scheduler.step()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        logger.done()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        logger.done()
        raise e
    else:
        logger.done()

if __name__ == "__main__":
    train_model() 