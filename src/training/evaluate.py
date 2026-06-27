import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from models.model_architecture import DeepfakeDetector

# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")
BATCH_SIZE = 8
IMAGES_PER_CLASS = 500   # 🔥 safe & fast for evaluation

print("🚀 Evaluation started...", flush=True)

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- LOAD FULL TEST DATASET ----------------
print("📂 Loading test dataset...", flush=True)

full_test_dataset = datasets.ImageFolder(
    root="dataset/real_vs_fake/real-vs-fake/test",
    transform=transform
)

print(f"📊 Total test images found: {len(full_test_dataset)}", flush=True)
print(f"📁 Classes: {full_test_dataset.classes}", flush=True)

# ---------------- FAST SUBSET SELECTION ----------------
print("⚡ Selecting evaluation subset...", flush=True)

samples = full_test_dataset.samples  # (path, label)
class_indices = {0: [], 1: []}

for idx, (_, label) in enumerate(samples):
    class_indices[label].append(idx)

for label in class_indices:
    np.random.shuffle(class_indices[label])

selected_indices = (
    class_indices[0][:IMAGES_PER_CLASS] +
    class_indices[1][:IMAGES_PER_CLASS]
)

test_dataset = Subset(full_test_dataset, selected_indices)

print(f"✅ Using {len(test_dataset)} images for evaluation", flush=True)

# ---------------- DATALOADER ----------------
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ---------------- LOAD MODEL ----------------
print("🧠 Loading trained model...", flush=True)

model = DeepfakeDetector()
model.load_state_dict(
    torch.load("models/best_deepfake_model.pth", map_location=DEVICE)
)
model = model.to(DEVICE)
model.eval()

# ---------------- EVALUATION ----------------
all_preds = []
all_labels = []

print("📈 Running inference...", flush=True)

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(DEVICE)
        outputs = model(images)
        outputs = torch.sigmoid(outputs)

        preds = (outputs >= 0.5).int().cpu().numpy().flatten()
        preds = 1 - preds
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

        if batch_idx % 10 == 0:
            print(f"   Processed batch {batch_idx}", flush=True)

# ---------------- METRICS ----------------
print("\n📊 Evaluation Results", flush=True)
print("----------------------", flush=True)

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"Accuracy  : {accuracy:.4f}", flush=True)
print(f"Precision : {precision:.4f}", flush=True)
print(f"Recall    : {recall:.4f}", flush=True)
print(f"F1‑Score  : {f1:.4f}", flush=True)

print("\nConfusion Matrix:", flush=True)
print(cm, flush=True)
print(full_test_dataset.classes)
