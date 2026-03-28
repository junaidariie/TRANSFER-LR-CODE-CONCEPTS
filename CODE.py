# 1) Replace head
in_features = self.model.fc.in_features
self.model.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes)
)

# 2) Freeze everything
for p in self.model.parameters():
    p.requires_grad = False

# 3) Unfreeze head
for p in self.model.fc.parameters():
    p.requires_grad = True

# 4) 🔥 Unfreeze last block (layer4)
for p in self.model.layer4.parameters():
    p.requires_grad = True
    
#===================================================================================

class RandomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Load pretrained model
        self.model = models.resnet18(weights="IMAGENET1K_V1")

        # Replace classification fc
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # ---- Freeze everything ----
        for param in self.model.parameters():
            param.requires_grad = False

        # ---- Always train fc ----
        for param in self.model.fc.parameters():
            param.requires_grad = True
            
        # ----- here can unfreeze some last layers ---
        
        # ---- Gradual unfreezing (controlled) ----
        if unfreeze_last_n > 0:
            layers = list(self.model.children())

            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True


    def forward(self, x):
        return self.model(x)
