import streamlit as st
import torch
#from transformers import ViTImageProcessor, ViTForImageClassification
import transformers as tf
#from PIL import Image
import PIL as pil

#load model and processor
@st.cache_resource
def load_model():
    model_name = "google/vit-base-patch16-224"
    processor = tf.ViTImageProcessor.from_pretrained(model_name)
    model = tf.ViTForImageClassification.from_pretrained(model_name)
    return processor,model


#predicion function
def classify(image,processor,model):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    #print("Predicted class:", model.config.id2label[predicted_class_idx])
    return model.config.id2label[predicted_class_idx]


#steamlit UI
st.title("Hugging face image classifier google/vit-base-patch16-224")
uploaded_file = st.file_uploader("upload image",type=["jpg","png","jpeg"])
#path error resolution - Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
torch.classes.__path__ = []

if uploaded_file:
    #get image
    image = pil.Image.open(uploaded_file).convert("RGB")
    st.image(image,caption="uploaded image",use_container_width=True)

    #call prediction functions
    processor,model = load_model()
    prediction = classify(image,processor,model)

    #display predicted results
    st.subheader(f"prediction is : {prediction}")
