class CustomEfficientNet(nn.Module):
    # 2. I am not sure but why is config class a type when I check type(config)
    def __init__(self, config: type, pretrained: bool = True):
        super().__init__()
        self.config = config
        # For myself, I like to set argument names for each
        self.model = geffnet.create_model(
            model_weight_path=config.model_weight_path, model_name=config.effnet, pretrained=pretrained
        )
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, config.num_classes)

    def forward(self, x):
        # TODO: add dropout layers, or the likes.
        x = self.model(x)
        return x