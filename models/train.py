import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from tqdm import tqdm
from data.preprocess import get_train_data_loaders
import utils.utils as ut
from models.stock_model import get_stock_model
from models.inception_resnet import InceptionResNet
from medmnist import INFO
import pandas as pd
import time
import argparse

class MyModel:
    def __init__(self, model_id, data_flag_id = 0, lr_strategy_id = 0, if_pretrained = 'y', scheduler_id=0, epochs=30, batch_size=32):
        
        MODEL_DICT = ut.get_model_dict()
        if model_id not in MODEL_DICT.keys():
            raise ValueError(f"Unsupported model, please use -h for help.")
        else:
            self.model_name = MODEL_DICT[model_id]
        
        DATA_DICT = ut.get_data_dict()
        if data_flag_id not in DATA_DICT.keys():
            raise ValueError(f"Unsupported data, please use -h for help.")
        else:
            self.data_flag = DATA_DICT[data_flag_id]

        PRETRAIN_DICT = ut.get_if_pretrain()
        if if_pretrained not in PRETRAIN_DICT.keys():
            raise ValueError(f"Unsupported pretrain mode, please use -h for help.")
        else:
            self.pretrained = PRETRAIN_DICT[if_pretrained]
        
        SCHEDULAR_DICT = ut.get_lr_schedular()
        if scheduler_id not in SCHEDULAR_DICT.keys():
            raise ValueError(f"Unsupported Schedular, please use -h for help.")
        else:
            self.scheduler_type = SCHEDULAR_DICT[scheduler_id]
        
        LR_DICT = ut.get_lr_strategy()
        if lr_strategy_id not in LR_DICT.keys():
            try:
                self.lr = float(lr_strategy_id)
            except:
                raise ValueError(f"Invalid learning rate strategy. Choose from {LR_DICT.keys()} or input a number.")
        else:
            self.lr = LR_DICT[lr_strategy_id]

        self.num_classes = len(INFO[self.data_flag]['label'])
        self.epochs = epochs
        self.batch_size = batch_size
        if self.batch_size not in [8, 16, 32, 64, 128]:
            raise ValueError(f"Unsupported batch size, please use -h for help.")
        self.__create_output_folder()
    
    def __create_output_folder(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Get the directory of the current script
        current_dir = os.path.abspath(os.path.dirname(__file__))
        output_folder = os.path.abspath(os.path.join(current_dir, "../outputs"))
        self.run_folder = os.path.join(output_folder, f"{self.data_flag}_{self.model_name.replace('_', '')}_{timestamp}")
        os.makedirs(self.run_folder, exist_ok=True)


    def train_model(self):
        # Select model
        if self.model_name != 'inception_resnet':
            model = get_stock_model(self.model_name, self.num_classes, self.pretrained)
        else:
            print("Pretrained model weight is not available for InceptionResnet.")
            model = InceptionResNet(self.num_classes)
            self.pretrained = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Load data
        train_loader, val_loader = get_train_data_loaders(data_flag=self.data_flag, batch_size=self.batch_size)
        # first_batch = next(iter(train_loader))
        # labels = first_batch[1]  # Example labels from the first batch

        # Determine single-label or multi-label
        info = INFO[self.data_flag]
        is_multi_label = info['task'] == "multi-label, binary-class"

        # Calculate class weights
        all_labels = torch.cat([batch[1] for batch in train_loader], dim=0) # Shape [N, num_classes] or Shape [N]

        # Define loss function
        if is_multi_label:
            class_weights = 1.0 / (all_labels.sum(dim=0) + 1e-6)  # Compute for each class
            class_weights = class_weights.to(device)  # Move to device
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            all_labels = all_labels.view(-1).long()  # Ensure 1D and integer type
            class_counts = torch.bincount(all_labels, minlength=self.num_classes)  # Class counts
            class_weights = 1.0 / (class_counts + 1e-6)  # Inverse frequency
            class_weights = class_weights / class_weights.sum()  # Normalize weights (optional)
            class_weights = class_weights.to(device)  # Move to device
            
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Define learning rate scheduler
        scheduler = None
        if self.scheduler_type == "exponential":
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.scheduler_type == "ratio_decrease":
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 if epoch > 5 else 1.0)

        # Logging setup
        log_file = os.path.join(self.run_folder, f"{self.model_name}_training_log.csv")
        logs = []

        best_val_loss = float("inf")
        best_model_path = os.path.join(self.run_folder, f"{self.model_name}_best_model.pth")

        # Training loop
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                if is_multi_label:
                    labels = labels.float()  # Convert to float for BCEWithLogitsLoss
                else:
                    labels = labels.squeeze().long()   # Convert to long for CrossEntropyLoss

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            # Validation loop
            model.eval()
            val_loss = 0.0
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    if is_multi_label:
                        labels = labels.float()  # Convert to float for BCEWithLogitsLoss
                    else:
                        labels = labels.squeeze().long()   # Convert to long for CrossEntropyLoss

                    loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    if not is_multi_label:  # Accuracy only applies to single-label
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = correct / total if not is_multi_label else None

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}")

            logs.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy if val_accuracy else "N/A"
            })
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy if val_accuracy else 'N/A'}")

            if scheduler:
                scheduler.step()
                print(f"Epoch {epoch+1}: Learning Rate = {scheduler.get_last_lr()[0]}")
        # Save logs
        log_df = pd.DataFrame(logs)
        ut.plot_training_log(log_df, self.run_folder)
        log_df.to_csv(log_file, index=False)
        print(f"Training log saved to {log_file}")
        params = {
            "model_name": self.model_name,
            "lr_strategy": self.lr,
            "pretrained": self.pretrained,
            "scheduler_type": self.scheduler_type if self.scheduler_type else "None",
            "batch_size": self.batch_size,
            "epochs": self.epochs
        }
        params_df = pd.DataFrame([params])
        params_file = os.path.join(self.run_folder, "training_parameters.csv")    
        params_df.to_csv(params_file, index=False)
        print(f"Training parameters saved to {params_file}")


        return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type = int, required = True, help = f"Select Model from {ut.get_model_dict()} for training")
    parser.add_argument("-d", type= int, default = 0, help = f"Select Data from {ut.get_data_dict()} for training. pathmnist (0) is default")
    parser.add_argument("-l", type= int, default= 0, help = f"Choose the start learning rate from {ut.get_lr_strategy()}, or use a float number.")
    parser.add_argument("-p", type = str, default = 'y', help = f"Choose if use pretrain weights: {ut.get_if_pretrain()}")
    parser.add_argument('-s', type = int, default = 0, help = f"Choose learning rate schedular from {ut.get_lr_schedular()}")
    parser.add_argument('-e', type = int, default = 30, help = "Set the number of epochs (default = 30).")
    parser.add_argument('-b', type = int, default = 32, help = "Batchsize from [8, 16, 32, 64, 128]")
    args = parser.parse_args()

    model = MyModel(model_id = args.m, data_flag_id = args.d, lr_strategy_id=args.l, if_pretrained = args.p, scheduler_id=args.s, epochs=args.e, batch_size=args.b)
    model.train_model()