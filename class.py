import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# Streamlit code
st.title('Image Classification with Vision Transformer')

# Load the model and processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    st.write("Classifying...")
    # Call the prediction function
    try:
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        st.write("Predicted class:", model.config.id2label[predicted_class_idx])
    except Exception as e:
        st.write("An error occurred:", e)
