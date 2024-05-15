
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

class DeepFeatureExtractor:
    def __init__(self):
        self.model = resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))  # Remove classification layer
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract_features(self, normalized_iris):
        """
        Extract deep features using a pretrained ResNet model.
        Args:
        - normalized_iris (np.ndarray): Normalized iris image.
        Returns:
        - deep_features (torch.Tensor): Extracted deep features.
        """
        image = self.transform(normalized_iris).unsqueeze(0)
        with torch.no_grad():
            deep_features = self.model(image).flatten()
        return deep_features
