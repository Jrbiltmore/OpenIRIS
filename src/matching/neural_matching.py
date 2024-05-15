
import torch
import torch.nn as nn

class NeuralMatcher(nn.Module):
    def __init__(self):
        super(NeuralMatcher, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, feature1, feature2):
        """
        Forward pass for matching features using a neural network.
        Args:
        - feature1 (torch.Tensor): First feature vector.
        - feature2 (torch.Tensor): Second feature vector.
        Returns:
        - similarity (float): Similarity score.
        """
        x = torch.cat((feature1, feature2), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        similarity = torch.sigmoid(self.fc3(x))
        return similarity
