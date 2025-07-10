# 🔧 Install dependencies (if not done)
!pip install insightface onnxruntime opencv-python matplotlib -q

# 📌 Import modules
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from google.colab import files
from insightface.app import FaceAnalysis

# ✅ Initialize InsightFace model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# ✅ Set path to your larger dataset
KNOWN_DIR = '/content/drive/MyDrive/dataset/dataset'
known_faces = {}

# 📌 Function to extract face embedding
def extract_feature(path):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Failed to read image: {path}")
        return None
    faces = app.get(img)
    if faces:
        return faces[0].normed_embedding
    else:
        print(f"⚠️ No face found in image: {path}")
    return None

# 🔁 Load known student faces
for filename in os.listdir(KNOWN_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(KNOWN_DIR, filename)
        name = os.path.splitext(filename)[0]
        emb = extract_feature(path)
        if emb is not None:
            known_faces[name] = (path, emb)

print(f"\n✅ Loaded {len(known_faces)} known students from full dataset.\n")

# 📤 Upload your test image
uploaded = files.upload()
test_image_path = list(uploaded.keys())[0]

# 🔍 Matching function
def match_image(test_path, known_faces, threshold=1.2):
    test_feat = extract_feature(test_path)
    if test_feat is None:
        print("❌ No face detected in test image.")
        return "Unknown", None, None

    best_name, best_path, best_dist = "Unknown", None, float("inf")
    print(f"\n👉 Matching against {len(known_faces)} known faces...\n")

    for name, (path, emb) in known_faces.items():
        if emb is None:
            continue
        dist = norm(test_feat - emb)
        print(f"🔍 {name}: Distance = {dist:.3f}")
        if dist < best_dist and dist < threshold:
            best_name = name
            best_path = path
            best_dist = dist

    return best_name, best_path, best_dist if best_name != "Unknown" else None

# 📈 Show visual result
def show_result(test_path, matched_path, matched_name):
    img1 = cv2.cvtColor(cv2.imread(test_path), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(matched_path), cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(img1)
    axs[0].set_title("Test Image")
    axs[0].axis('off')

    axs[1].imshow(img2)
    axs[1].set_title(f"Matched: {matched_name}")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

# 🚀 Match and display result
name, matched_path, dist = match_image(test_image_path, known_faces)
print(f"\n✅ Matched: {name} (Distance: {dist})")

if matched_path:
    show_result(test_image_path, matched_path, name)
else:
    print("❌ No matching student found.")
