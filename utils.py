from prettytable import PrettyTable

class TrainingLogger:
    def __init__(self):
        self.header_printed = False
        
    def log(self, epoch, step, total_steps, train_acc, test_acc, loss, lr, time_taken):
        if not self.header_printed:
            print("+-------+---------+-----------+----------+--------+----------+---------+")
            print("| Epoch |   Step  | Train Acc | Test Acc |  Loss  |    LR    |  Time   |")
            print("+-------+---------+-----------+----------+--------+----------+---------+")
            self.header_printed = True
            
        test_acc_str = f"{test_acc:6.2f}%" if test_acc is not None else "   -    "
        print(f"|  {epoch:3d}  | {step:3d}/{total_steps:3d} |  {train_acc:6.2f}%  | {test_acc_str} | {loss:.4f} | {lr:.6f} | {time_taken:.1f}s |") 