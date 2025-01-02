from prettytable import PrettyTable

class TrainingLogger:
    def __init__(self):
        self.table = PrettyTable()
        self.table.field_names = ["Epoch", "Step", "Train Acc", "Test Acc", "Loss", "LR"]
        self.table.float_format = '.4'
        
        # For storing best metrics
        self.best_train_acc = 0
        self.best_test_acc = 0
        self.best_loss = float('inf')
    
    def log_step(self, epoch, step, train_acc, loss, lr, total_steps):
        """Log metrics during training steps"""
        # Clear previous step logs within same epoch
        if len(self.table.rows) > 0 and self.table.rows[-1][0] == epoch:
            self.table.del_row(-1)
            
        self.table.add_row([
            epoch,
            f"{step}/{total_steps}",
            f"{train_acc:.2f}%",
            "-",
            f"{loss:.4f}",
            f"{lr:.6f}"
        ])
        print(self.table)
        print("\033[F" * (len(self.table.rows) + 4))  # Move cursor up
        
    def log_epoch(self, epoch, train_acc, test_acc, loss, lr, total_steps):
        """Log metrics at the end of epoch"""
        # Update best metrics
        self.best_train_acc = max(self.best_train_acc, train_acc)
        self.best_test_acc = max(self.best_test_acc, test_acc)
        self.best_loss = min(self.best_loss, loss)
        
        # Add row with epoch summary
        self.table.add_row([
            epoch,
            f"{total_steps}/{total_steps}",
            f"{train_acc:.2f}%",
            f"{test_acc:.2f}%",
            f"{loss:.4f}",
            f"{lr:.6f}"
        ])
        print(self.table)
        print(f"\nBest Metrics:")
        print(f"Train Acc: {self.best_train_acc:.2f}%")
        print(f"Test Acc: {self.best_test_acc:.2f}%")
        print(f"Loss: {self.best_loss:.4f}")
        print("\033[F" * (len(self.table.rows) + 8))  # Move cursor up

    def done(self):
        """Print final metrics"""
        print("\n" * (len(self.table.rows) + 8))  # Move cursor down
        print(self.table)
        print(f"\nBest Metrics:")
        print(f"Train Acc: {self.best_train_acc:.2f}%")
        print(f"Test Acc: {self.best_test_acc:.2f}%")
        print(f"Loss: {self.best_loss:.4f}") 