import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.out_channels = out_channels * 3  # Total channels after concatenation
    
    def forward(self, x):
        branch1_out = self.branch1(x)
        branch3_out = self.branch3(x)
        branch5_out = self.branch5(x)
        return torch.cat([branch1_out, branch3_out, branch5_out], dim=1)

class InceptionResNet(nn.Module):
    def __init__(self, num_classes):
        super(InceptionResNet, self).__init__()
        self.stem = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        
        # Inception Blocks with increasing channels
        self.inception1 = InceptionBlock(64, 32)
        self.inception2 = InceptionBlock(96, 64)
        self.residual1 = nn.Conv2d(192, 192, kernel_size=1)
        
        self.inception3 = InceptionBlock(192, 64)
        self.inception4 = InceptionBlock(192, 128)
        self.residual2 = nn.Conv2d(384, 384, kernel_size=1)
        
        self.inception5 = InceptionBlock(384, 128)
        self.inception6 = InceptionBlock(384, 192)
        self.residual3 = nn.Conv2d(576, 576, kernel_size=1)
        
        # Final global pooling and classification layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(576, num_classes)
    
    def forward(self, x):
        # Stem layer
        x = F.relu(self.stem(x))
        
        # Inception blocks with residual connections
        x = F.relu(self.inception1(x))
        x = F.relu(self.inception2(x))
        x = x + self.residual1(x)  # Residual connection
        
        x = F.relu(self.inception3(x))
        x = F.relu(self.inception4(x))
        x = x + self.residual2(x)  # Residual connection
        
        x = F.relu(self.inception5(x))
        x = F.relu(self.inception6(x))
        x = x + self.residual3(x)  # Residual connection
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def load_pretrained_weights(self, weight_file):
        """
        Load weights from a pretrained file.

        :param weight_file: Path to the weight file.
        """
        try:
            state_dict = torch.load(weight_file)
            self.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from {weight_file}")
        except Exception as e:
            print(f"Failed to load pretrained weights. Error: {e}")
