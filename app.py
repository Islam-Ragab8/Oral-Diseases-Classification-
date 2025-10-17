# app.py

import streamlit as st
from src.utils.inference import predict_images
from src.utils.config import config
from PIL import Image
import tempfile
import os

# ---------- إعداد الصفحة ----------
st.set_page_config(
    page_title="🦷 Oral Diseases Classification",
    page_icon="🦷",
    layout="wide"
)

st.title("🦷 Oral Diseases Classification App")
st.write("ارفع صورة أو أكثر، والموديل هيحاول يتعرف على نوع المرض.")

# ---------- رفع الصور ----------
uploaded_files = st.file_uploader(
    "اختار صورة أو أكثر:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ---------- عند رفع الصور ----------
if uploaded_files:
    st.info(f"📸 تم رفع {len(uploaded_files)} صورة.")
    temp_paths = []

    # حفظ الصور مؤقتًا
    for file in uploaded_files:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(file.read())
        temp_paths.append(temp_file.name)

    # ---------- تنفيذ التنبؤ ----------
    try:
        results = predict_images(temp_paths)
        st.success("✅ تم تحليل الصور بنجاح!")

        # ---------- عرض النتائج ----------
        for img_path, prediction in zip(temp_paths, results.predictions):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(Image.open(img_path), caption=prediction.base_name, use_container_width=True)
            with col2:
                st.markdown(f"**🩺 Diagnosis:** `{prediction.class_name}`")
                st.markdown(f"**📊 Confidence:** `{prediction.confidence:.2f}`")
                st.markdown(f"**🔢 Class Index:** `{prediction.class_index}`")
            st.divider()

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

    # ---------- تنظيف الملفات المؤقتة ----------
    for path in temp_paths:
        try:
            os.remove(path)
        except:
            pass

else:
    st.info("⬆️ ارفع صورة علشان تبدأ التحليل.")
