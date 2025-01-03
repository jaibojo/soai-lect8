import torch
import torch.optim as optim
import torch.nn as nn
import math
import time
from torchvision import transforms, datasets
from model import CIFAR10Model
from utils import TrainingLogger
from torch.optim.lr_scheduler import LambdaLR
from dataset import CIFAR10Transform, CIFAR10Dataset


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        # Longer cycles for more stable learning
        cycle_length = 40
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        
        # Get current position in cycle
        cycle = (epoch - warmup_epochs) // cycle_length
        cycle_epoch = (epoch - warmup_epochs) % cycle_length
        
        # Cosine decay with higher minimum LR
        cosine_decay = 0.5 * (1 + math.cos(math.pi * cycle_epoch / cycle_length))
        
        # Three-phase learning within cycle
        if cycle_epoch < cycle_length * 0.3:  # First 30% - aggressive learning
            lr = 0.15 + 0.85 * cosine_decay
        elif cycle_epoch < cycle_length * 0.7:  # Middle 40% - balanced learning
            lr = 0.1 + 0.6 * cosine_decay
        else:  # Last 30% - fine-tuning
            lr = 0.075 + 0.325 * cosine_decay
        
        # More gradual LR reduction between cycles
        lr *= 0.9 ** cycle
        
        return lr
    
    return LambdaLR(optimizer, lr_lambda)


def train_model():
    # CIFAR-10 mean and std values
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    print("Preparing datasets...")
    
    # Use custom transforms from dataset.py
    train_transform = CIFAR10Transform(mean=mean, std=std, train=True)
    test_transform = CIFAR10Transform(mean=mean, std=std, train=False)
    
    trainset = CIFAR10Dataset(
        root='./data', 
        train=True,
        transform=train_transform,
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
        transform=test_transform,
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
        lr=0.002,
        weight_decay=6e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    num_epochs = 120  # Total epochs
    warmup_epochs = max(1, int(0.05 * num_epochs))  # Dynamic warmup (5% of total epochs)
    print(f"Using {warmup_epochs} warmup epochs")
    
    scheduler = get_lr_scheduler(optimizer, warmup_epochs, num_epochs)
    
    logger = TrainingLogger()
    print("Starting training...")
    
    # Track best accuracy
    best_acc = 0.0
    
    try:
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()  # Start timing the epoch
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Training loop
            for i, (inputs, labels) in enumerate(trainloader, 1):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if i % 50 == 0 or i == len(trainloader):
                    train_loss = running_loss / i
                    train_acc = 100. * correct / total
                    epoch_time = time.time() - epoch_start_time  # Calculate time taken
                    logger.log(epoch, i, len(trainloader), train_acc, None, train_loss, scheduler.get_last_lr()[0], epoch_time)
            
            # Validation
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            test_loss = test_loss / len(testloader)
            test_acc = 100. * correct / total
            epoch_time = time.time() - epoch_start_time  # Calculate total epoch time
            
            # Log final stats for the epoch
            logger.log(epoch, len(trainloader), len(trainloader), train_acc, test_acc, train_loss, scheduler.get_last_lr()[0], epoch_time)
            
            scheduler.step()
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), 'best_model.pth')
                
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
    except Exception as e:
        print('\nError during training:', str(e))
        raise e
    
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train_model()
