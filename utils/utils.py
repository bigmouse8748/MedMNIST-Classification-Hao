import matplotlib.pyplot as plt
import os


def get_model_dict():
    model_dict = {0: 'resnet50',
                  1: 'resnet18',
                  2: 'efficientnet_b0',
                  3: 'mobilenet_v3_large',
                  4: 'inception_resnet'
                }
    return model_dict

def get_model_name_dict():
    model_name_dict = {"inceptionresnet": "inception_resnet",
                        "resnet50": "resnet50",
                        "resnet18": "resnet18",
                        "efficientnetb0": "efficientnet_b0",
                        "mobilenetv3large": "mobilenet_v3_large"
                        }
    return model_name_dict

def get_data_dict():
    data_dict = {0: 'pathmnist', 
                 1: 'chestmnist', 
                 2: 'dermamnist',
                 3: 'octmnist', 
                 4: 'pneumoniamnist', 
                 5: 'retinamnist', 
                 6: 'breastmnist', 
                 7: 'bloodmnist', 
                 8: 'tissuemnist',
                 9: 'organamnist', 
                 10: 'organcmnist', 
                 11: 'organsmnist'
                 }
    return data_dict

def get_lr_strategy():
    lr_options = {0: 0.001, 
                  1: 0.0001, 
                  2: 0.01}
    return lr_options

def get_if_pretrain():
    pretrain = {"y": True,
                "n": False}
    return pretrain

def get_lr_schedular():
    schedular = {0: None, 
                 1: 'exponential', 
                 2: 'ratio_decrease'}
    return schedular

def plot_training_log(df, save_dir):
    # Create a double y-axis plot
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot train_loss and val_loss on primary y-axis
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="tab:blue", marker="o")
    ax1.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="tab:cyan", marker="s")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    # Create secondary y-axis for validation accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation Accuracy", color="tab:red")
    ax2.plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy", color="tab:red", marker="^")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Title and grid
    plt.title("Training Loss, Validation Loss, and Validation Accuracy")
    ax1.grid(True)

    # Show plot
    plt.savefig(os.path.join(save_dir, "training_log_plot.png"))