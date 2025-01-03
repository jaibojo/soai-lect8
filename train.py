import torch
import torch.optim as optim
import torch.nn as nn
import math
from torchvision import transforms, datasets
from model import CIFAR10Model
from utils import TrainingLogger
from torch.optim.lr_scheduler import LambdaLR


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        # Restart every 25 epochs (instead of 20)
        cycle_length = 25
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        
        # Get current position in cycle
        cycle = (epoch - warmup_epochs) // cycle_length
        cycle_epoch = (epoch - warmup_epochs) % cycle_length
        
        # Cosine decay with minimum LR of 1% of max
        cosine_decay = 0.5 * (1 + math.cos(math.pi * cycle_epoch / cycle_length))
        lr = 0.01 + 0.99 * cosine_decay  # Decay from 1.0 to 0.01
        
        # Reduce max LR by 10% each cycle
        lr *= 0.9 ** cycle
        
        return lr
    
    return LambdaLR(optimizer, lr_lambda)


def train_model():
    # CIFAR-10 mean and std values
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    print("Preparing datasets...")
    
    # Training transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3),
            value=0
        )
    ])
    
    # Test transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    trainset = datasets.CIFAR10(
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

    testset = datasets.CIFAR10(
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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    num_epochs = 100
    warmup_epochs = 5
    scheduler = get_lr_scheduler(optimizer, warmup_epochs, num_epochs)
    
    logger = TrainingLogger()
    print("Starting training...")
    
    # Track best accuracy
    best_acc = 0.0
    
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
                
                # Clip gradients after backward pass
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                print(f"\nNew best accuracy: {best_acc:.2f}%")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                }, "best_model.pth")
            
            logger.log_epoch(epoch, train_acc, test_acc, avg_loss, current_lr, len(trainloader))
            scheduler.step()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user, saving model checkpoint...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, "model_checkpoint.pth")
        logger.done()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        logger.done()
        raise e
    else:
        print(f"\nTraining completed. Best accuracy: {best_acc:.2f}%")
        logger.done()


if __name__ == "__main__":
    train_model()
