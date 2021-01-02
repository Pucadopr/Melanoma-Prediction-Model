from torch import nn
import geffnet

class CustomEfficientNet(nn.Module):
    
    def __init__(self, config: type, pretrained: bool = True):
        super().__init__()
        self.config = config

        self.model = geffnet.create_model(
            model_weight_path=config.model_weight_path, model_name=config.effnet, pretrained=pretrained
        )
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, config.num_classes)

    def forward(self, x):
        x = self.model(x)
        return x