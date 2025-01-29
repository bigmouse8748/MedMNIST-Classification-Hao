import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
import torch
import pandas as pd
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from medmnist import INFO
from data.preprocess import get_test_data_loader  # Assuming preprocess.py has a get_test_data_loader function
from models.stock_model import get_stock_model
from models.inception_resnet import InceptionResNet
import utils.utils as ut
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse


class ModelTester:

    def __init__(self, data_flag_id = "pathmnist", threshold=0.5):
        DATA_DICT = ut.get_data_dict()
        if data_flag_id not in DATA_DICT.keys():
            raise ValueError(f"Unsupported data, please use -h for help.")
        else:
            self.data_flag = DATA_DICT[data_flag_id]

        if threshold <= 0 or threshold >= 1:
            raise ValueError(f"Please set the threshold between 0 and 1.")
        else: self.threshold = threshold

        self.output_folder = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../outputs"))
        self.test_result_folder = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_results"))
        os.makedirs(self.test_result_folder, exist_ok=True)
        self.batch_size = 32
        self.threshold = threshold

        # Load dataset info to determine num_classes and task type
        info = INFO[self.data_flag]
        self.num_classes = len(info['label'])

        self.classes = info["label"]
        self.is_multi_label = info['task'] == "multi-label, binary-class"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name_dict = ut.get_model_name_dict()

    def load_model(self, model_path, model_name):
        """
        Load a model from a given folder.

        Args:
            model_path (str): Path to the model folder.
            model_name (str): Name of the model.

        Returns:
            torch.nn.Module: The loaded model.
        """
        # Load model based on the folder name

        if model_name == "inception_resnet":
            model = InceptionResNet(num_classes=self.num_classes)
        else:
            model = get_stock_model(model_name, num_classes=self.num_classes, pretrained=False)
        model_full_name = f"{model_name}_best_model.pth"

        model_path = os.path.join(model_path, model_full_name)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def plot_multiclass_roc_auc(self, y_true, y_pred, save_folder):
        """
        Plots the ROC-AUC curve for a multi-class classification problem.
        
        Parameters:
        y_true: Ground truth labels (shape: [n_samples], integer class labels).
        y_pred: Predicted probabilities (shape: [n_samples, num_classes]).
        num_classes: Total number of classes.
        """
        # Binarize the true labels for OvR
        y_true_binarized = label_binarize(y_true, classes=np.arange(self.num_classes))
        y_pred_binarized = label_binarize(y_pred, classes=np.arange(self.num_classes))
        # Compute ROC curve and ROC AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}

        if self.num_classes > 2:
            for i in range(self.num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        

            # Compute micro-average ROC curve and AUC
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_pred_binarized.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Compute macro-average ROC curve and AUC
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.num_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(self.num_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= self.num_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot all ROC curves
            plt.figure(figsize=(10, 8))
            plt.plot(fpr["micro"], tpr["micro"],
                    label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
                    color='deeppink', linestyle=':', linewidth=4)

            plt.plot(fpr["macro"], tpr["macro"],
                    label=f'macro-average ROC curve (area = {roc_auc["macro"]:.2f})',
                    color='navy', linestyle=':', linewidth=4)

            colors = plt.cm.get_cmap('tab10', self.num_classes)
            for i, color in zip(range(self.num_classes), colors(range(self.num_classes))):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'Class {self.classes[str(i)]} (area = {roc_auc[i]:.2f})')

        else:
            fpr, tpr, _ = roc_curve(y_true_binarized, y_pred_binarized)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='r', lw=2, label=f'area = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f"Receiver Operating Characteristic (ROC) Curve with threshold {self.threshold}", fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid()
        plt.savefig(f"{save_folder}/roc_auc_{self.threshold}.png", dpi = 300)
        plt.show()

        return roc_auc
    
    def get_metrics(self, all_targets, all_predictions):
         # Compute metrics
        metrics = {}
        if self.is_multi_label:
            metrics["hamming_loss"] = hamming_loss(all_targets, all_predictions)
            metrics["exact_match_ratio"] = accuracy_score(all_targets, all_predictions)
        else:
            metrics["hamming_loss"] = hamming_loss(all_targets, all_predictions)
            metrics["exact_match_ratio"] = accuracy_score(all_targets, all_predictions)
            metrics["accuracy"] = accuracy_score(all_targets, all_predictions)
            metrics["precision_macro"] = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
            metrics["recall_macro"] = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
            metrics["f1_macro"] = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
            metrics["threshold"] = self.threshold

        return metrics


    def validate_model(self, model, test_loader):
        """
        Validate the model on the test dataset and compute metrics.

        Args:
            model: Trained PyTorch model.
            test_loader: DataLoader for test data.

        Returns:
            metrics: Dictionary containing validation metrics.
        """
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)

                # Apply sigmoid for multi-label, softmax for single-label
                if self.is_multi_label:
                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities > self.threshold).int()
                else:
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)

                all_targets.append(labels.cpu())
                all_predictions.append(predictions.cpu())

        # Concatenate all targets and predictions
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_predictions = torch.cat(all_predictions, dim=0).numpy()

        val_metrics = self.get_metrics(all_targets, all_predictions)
        return val_metrics

    def test_all_models(self):
        """
        Test all models in the output folder and print the results.
        """
        all_results_folder = os.path.join(self.test_result_folder, "Group_results")
        os.makedirs(all_results_folder, exist_ok=True)

        test_loader = get_test_data_loader(data_flag=self.data_flag, batch_size=self.batch_size)
        model_folders = [os.path.join(self.output_folder, folder) for folder in os.listdir(self.output_folder)
                         if folder.split("_")[0] == self.data_flag]

        results = []

        for folder in model_folders:
            model_name = folder.split("\\")[-1].split("_")[1]
            model_full_name = f"{model_name}_best_model.pth"
            datetime = folder.split("\\")[-1].split("_")[2]
            print(f"Testing model: {model_name} in folder: {folder}")
            best_model_path = os.path.join(folder, model_full_name)

            if not Path(best_model_path).exists():
                print(f"Best model not found in {folder}. Skipping...")
                continue

            # Load model and validate
            model = self.load_model(folder, model_name)
            metrics = self.validate_model(model, test_loader)

            # Log metrics
            metrics["model_name"] = model_full_name
            metrics["model_time"] = datetime
            metrics["folder"] = folder
            results.append(metrics)

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_csv = os.path.join(all_results_folder, f"test_results_{self.data_flag}_{self.threshold}.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"Results saved to {results_csv}")

        return results
    
    def test_single_model(self, folder):
        '''
        folder: the folder dir that saves the model.pth
        '''
        single_results_folder = os.path.join(self.test_result_folder, folder.split('\\')[-1])
        os.makedirs(single_results_folder, exist_ok = True)
        if folder.split('\\')[-1].split('_')[0] != self.data_flag:
            print("The model does not fit the data_flag you selected.")
        test_loader = get_test_data_loader(data_flag=self.data_flag, batch_size=self.batch_size)
        model_name = self.model_name_dict[folder.split("\\")[-1].split("_")[1]]
        
        datetime = folder.split("\\")[-1].split("_")[2]
        print(f"Testing model: {model_name} in folder: {folder}")

        # Load model and validate
        model = self.load_model(folder, model_name)
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)

                # Apply sigmoid for multi-label, softmax for single-label
                if self.is_multi_label:
                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities > self.threshold).int()
                else:
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)

                all_targets.append(labels.cpu())
                all_predictions.append(predictions.cpu())

        # Concatenate all targets and predictions
        if not self.is_multi_label:
            all_targets = torch.cat(all_targets, dim=0).numpy().reshape(1, -1)[0]
        else:
            all_targets = torch.cat(all_targets, dim=0).numpy()
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        # all_predictions = np.array(all_predictions)
        if not self.is_multi_label:
            self.plot_multiclass_roc_auc(all_targets,all_predictions, single_results_folder)
        
        model_full_name = f"{model_name}_best_model.pth"
        single_metrics = self.get_metrics(all_targets, all_predictions)
        single_metrics["model_name"] = model_full_name
        single_metrics["model_time"] = datetime
        single_metrics["folder"] = folder

        # Save results to CSV
        results_df = pd.DataFrame([single_metrics])
        # results_csv = os.path.join(folder, f"test_results_{self.data_flag}.csv")
        results_csv = os.path.join(single_results_folder, f"test_results_{self.threshold}.csv")
        results_df.to_csv(results_csv, index = False)
        print(f"Results saved to {results_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type= int, default = 0, help = f"Select Data from {ut.get_data_dict()} for training. pathmnist (0) is default")
    parser.add_argument("-t", type= float, default = 0.5, help = "Set a threshold of probability for classification from 0 to 1")
    parser.add_argument("-m", type = int, default = 0, help = "Choose the test mode. [0: One model test (need to provide the path of output folder), 1: All model test]")
    args = parser.parse_args()
    test = ModelTester(data_flag_id=args.d, threshold = args.t)

    if args.m == 0:
        outputdir = str(input("Please input the output directory (such as ../outputs/chestmnist_inceptionresnet_20250128-234607):"))
        test.test_single_model(outputdir)
    elif args.m == 1:
        test.test_all_models()
    else:
        raise ValueError(f"Unsupported test mode, please use -h for help.")
