import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json

# ==============================
# 1) Load Model and Labels
# ==============================
MODEL_PATH = "C:\Users\7eGaZy CoMp\OneDrive\Desktop\Oral app\model.pth"         # Path to your trained model
LABELS_PATH = "C:\Users\7eGaZy CoMp\OneDrive\Desktop\Oral app\labels.json"      # Path to your labels file

# Load label dictionary
with open(LABELS_PATH, "r") as f:
    idx_to_class = json.load(f)  # Mapping from class index to disease name

# Load the trained model
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()

# ==============================
# 2) Define Image Transform
# ==============================
image_size = 224  # Should match the image size used during training
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# 3) Streamlit UI
# ==============================
st.set_page_config(page_title="Oral Diseases Classifier", page_icon="🦷")
st.title("🦷 Oral Diseases Classification")
st.write("Upload an image of the affected area and the model will predict the disease 👇")

# ==============================
# 4) Upload Image
# ==============================
uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0)

    # ==============================
    # 5) Prediction
    # ==============================
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = idx_to_class[str(pred_idx)]
        confidence = probs[0][pred_idx].item() * 100

    # ==============================
    # 6) Show Results
    # ==============================
    st.success(f"✅ Predicted disease: **{pred_label}**")
    st.info(f"📊 Confidence: {confidence:.2f}%")
