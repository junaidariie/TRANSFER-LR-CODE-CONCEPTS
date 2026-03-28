# TRANSFER-LR-CODE-CONCEPTS

## REPLACE THE HEAD FIRST OR JUST CHANGE OUT CLASS NUMBER

## THEN FREEZE THE INITIAL LAYERS

## THEN UNFREEZE HEAD/CLASSIFIER

## OPTIONAL : UNFREEZE FEW LAST LAYERS

```python
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
```

**Bold summary:** **Different models name their final classifier differently (`fc`, `head`, `classifier`) — use the attribute the model exposes, then choose one of four common fine‑tuning strategies: head‑only, last‑N layers, stagewise unfreeze, or full fine‑tune.** For ResNet use **`.fc`**, for many ViT/EfficientNet implementations use **`.head`**, and for VGG/AlexNet use **`.classifier`**.

### Quick comparison table of fine‑tuning strategies

| **Strategy** | **When to use** | **Typical attribute names** | **Pros** | **Cons** |
| --- | --- | --- | --- | --- |
| **Head only** | Small dataset; fast | `fc`; `head`; `classifier` | Fast; low overfitting risk | Limited adaptation |
| **Last‑N layers** | Moderate dataset; domain shift | same as above | Balances speed and adaptation | Need to pick N |
| **Stagewise unfreeze** | Larger dataset; careful tuning | same as above | Gradual adaptation; stable training | More complex schedule |
| **Full fine‑tune** | Large dataset; big domain gap | same as above | Maximum capacity to adapt | Slow; risk of overfitting |

---

### Concrete code patterns and order to follow

**1. Replace classifier then set `requires_grad`.**

**Best practice:** instantiate pretrained model → replace final layer → freeze backbone → enable grads for new head. This ensures the new head’s parameters are trainable. Example for a ViT/EfficientNet from `timm`:

```python
self.model = create_model('faster_vit_2_224', pretrained=True)
in_features = self.model.head.in_features
self.model.head = nn.Linear(in_features, num_classes)
for p in self.model.parameters(): p.requires_grad = False
for p in self.model.head.parameters(): p.requires_grad = True
```

This order is safe and common.

**2. Freeze everything then unfreeze last N layers**

```python
for p in model.parameters(): p.requires_grad = False
layers = list(model.children())
for layer in layers[-N:]:
    for p in layer.parameters(): p.requires_grad = True
```

Use this when you want to fine‑tune deeper features gradually.

**3. Stagewise unfreeze schedule**

- Train head only for a few epochs.
- Unfreeze last N layers and continue with a smaller LR.
- Optionally unfreeze more layers until full fine‑tune.
This reduces catastrophic forgetting and stabilizes training.

**4. Full fine‑tune**

Set all `requires_grad = True` and use a lower LR than training from scratch.

---

### Practical tips you must remember

- **Always inspect `print(model)`** to find the correct final layer name before replacing it.
- **Use different learning rates**: head LR ≈ (1\mathrm{e}{-3}), backbone LR ≈ (1\mathrm{e}{-4}) or lower.
- **Use optimizer param groups** to set separate LRs for head vs backbone.
- **Monitor overfitting** when unfreezing more layers; use weight decay and augmentation.

---

### Risks and tradeoffs

- **Head‑only**: fast but may underfit domain shifts.
- **Full fine‑tune**: best adaptation but needs more data and compute.
- **Stagewise**: safest but requires tuning schedule.

If you want, I can generate **ready‑to‑run templates** for ResNet, EfficientNet, ViT, and VGG showing exact attribute names and optimizer setups so you can copy‑paste them for future projects.

**Use the final-layer attribute the model exposes (`fc`, `head`, or `classifier`) and pick one of four fine‑tuning strategies (head‑only, last‑N, stagewise unfreeze, full). Below I show what your snippets do, why they differ by model, and concrete, copy‑paste examples for ResNet, EfficientNet/ViT, VGG with optimizer setups and LR tips.**

### Why `fc` vs `head` vs `classifier`

- **Different model implementations name the final classifier differently**: ResNets use **`fc`**, many ViT/EfficientNet implementations use **`head`**, and VGG/AlexNet use **`classifier`**. Always `print(model)` to confirm.

---

### What your code does explained

- 

```python
for param in self.model.parameters():
    param.requires_grad = False
```

## **Freezes the entire model** so no gradients are computed for backbone weights.

```python
for param in self.model.head.parameters():
    param.requires_grad = True
```

## **Enables training only for the head** (the new classifier). If you replaced the head after freezing, ensure you set `requires_grad=True` for the new head.

```python
for layer in layers[-unfreeze_last_n:]:
    for param in layer.parameters():
        param.requires_grad = True
```

**Unfreezes the last N layers** of the backbone for gradual fine‑tuning.

---

### Four common strategies with pros, cons, and code

| Strategy | When to use | Key code pattern | Pros | Cons |
| --- | --- | --- | --- | --- |
| **Head only** | Very small dataset | Freeze all; train head only | Fast; low overfit | Limited adaptation |
| **Last‑N layers** | Moderate data | Freeze all; unfreeze last N | Balance speed/adapt | Choose N carefully |
| **Stagewise unfreeze** | Medium→large data | Train head → unfreeze more gradually | Stable; avoids forgetting | More steps to manage |
| **Full fine‑tune** | Large dataset | All params trainable | Max adaptation | Slow; needs regularization |

---

### Concrete examples

### ResNet (torchvision)

```python
model = torchvision.models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
for p in model.parameters(): p.requires_grad = False
for p in model.fc.parameters(): p.requires_grad = True
```

**Use optimizer with head LR 1e-3.**

### EfficientNet / ViT (timm or similar)

```python
model = create_model('efficientnet_b0', pretrained=True)
in_f = model.head.in_features
model.head = nn.Linear(in_f, num_classes)
for p in model.parameters(): p.requires_grad = False
for p in model.head.parameters(): p.requires_grad = True
```

**Head LR 1e-3; backbone LR 1e-4 if unfreezing.**

### VGG / AlexNet

```python
model = torchvision.models.vgg16(pretrained=True)
in_f = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_f, num_classes)
```

---

### Optimizer param groups example

```python
optimizer = torch.optim.SGD([
    {'params': model.head.parameters(), 'lr': 1e-3},
    {'params': [p for n,p in model.named_parameters() if 'head' not in n and p.requires_grad], 'lr': 1e-4}
], momentum=0.9)
```

**Use lower LR for backbone when unfreezing.**

---

### Practical checklist to never forget

- **Always `print(model)`** to find final layer name.
- **Replace classifier before freezing** or re-enable `requires_grad` for the new layer.
- **Start with head-only for 3–5 epochs**, then unfreeze last‑N, reduce LR, continue.
- **Monitor validation loss** and use weight decay/augmentations to avoid overfitting.

If you want, I’ll generate ready‑to‑run templates for **ResNet50, EfficientNet B0, ViT, MobileNet, and VGG16** with training loops and optimizer schedules you can copy into your projects.

# ALSO IMPORTANT

---

# 🔹 Setup (common for all)

```python
import torch
import torch.nn as nn
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(weights="IMAGENET1K_V1")

# Replace classifier (VERY IMPORTANT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # example: 10 classes

model = model.to(device)
```

---

# 🔹 1. Full Fine-Tuning (Train Everything)

### 💡 When to use:

- Large dataset
- Domain very different from ImageNet

```python
# All params trainable (default)
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
```

### 🧠 Explanation:

- You’re **rewriting the entire network**
- High flexibility, high risk of overfitting
- Slowest training

👉 If your dataset is small, this is a bad idea.

---

# 🔹 2. Train Only Classifier (Feature Extractor)

### 💡 When to use:

- Small dataset
- Similar domain

```python
# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Only classifier learns
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
```

### 🧠 Explanation:

- Backbone acts as **fixed feature extractor**
- Only last layer adapts
- Fast + stable

👉 This is what beginners SHOULD start with.

---

# 🔹 3. Fine-Tune Last Few Layers (Hybrid Approach)

### 💡 When to use:

- Medium dataset
- Slight domain shift

```python
# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block + classifier
for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    list(model.layer4.parameters()) + list(model.fc.parameters()),
    lr=1e-4
)

criterion = nn.CrossEntropyLoss()
```

### 🧠 Explanation:

- Earlier layers = generic (edges, textures) → keep frozen
- Later layers = task-specific → adapt them
- Best tradeoff between speed and performance

👉 This is what actually wins in most real-world ML setups.

---

# 🔥 Critical Mistakes You Should Avoid

1. **Forgetting to freeze params**
    
    → then you're accidentally doing full training
    
2. **Wrong optimizer params**
    
    → optimizer should ONLY get trainable params
    
3. **Learning rate misuse**
    - Classifier only → higher LR (1e-3)
    - Fine-tuning → lower LR (1e-4 or less)
4. **Not replacing classifier**
    
    → model outputs 1000 classes = useless
    

---

# ⚡ Real Talk (Mentor Mode)

- If you’re just “trying all three,” you’re experimenting blindly
- You should **choose based on dataset size + similarity**

**Rule of thumb:**

- <5k images → classifier only
- 5k–50k → last layers
- 50k+ → full fine-tune

---

If you want, next step:

👉 I’ll show you how to structure this into your own reusable training library (the thing you were building earlier) so it looks like a mini-framework instead of random scripts.
