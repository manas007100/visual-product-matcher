import streamlit as st
import torch
from torchvision import models, transforms
import requests
from PIL import Image
import io
import numpy as np
from torch.nn import functional as F

# Load pre-trained model (cached for efficiency)
@st.cache_resource
def load_model():
    resnet = models.resnet50(weights='DEFAULT')
    resnet.eval()
    return torch.nn.Sequential(*list(resnet.children())[:-1])

feature_extractor = load_model()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extract feature vector from image
def get_feature(img):
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            feature = feature_extractor(input_tensor).squeeze()
        return feature
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Fetch products from public API (100000 products)
@st.cache_data
def fetch_products():
    try:
        response = requests.get('https://dummyjson.com/products?limit=100000')
        response.raise_for_status()
        return response.json()['products']
    except Exception as e:
        st.error(f"Error fetching products: {e}")
        return []

products = fetch_products()

# Precompute features with loading progress
@st.cache_data
def precompute_features(products):
    if not products:
        return []
    features = []
    progress_bar = st.progress(0)
    for i, p in enumerate(products):
        try:
            img_url = p['thumbnail']
            response = requests.get(img_url)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            feature = get_feature(img)
            features.append(feature)
        except:
            features.append(None)
        progress_bar.progress((i + 1) / len(products))
    return features

if products:
    with st.spinner('Precomputing product features (one-time)...'):
        product_features = precompute_features(products)
else:
    product_features = []

# Main app
st.title('Visual Product Matcher')

# Image input (file or URL)
upload_type = st.radio('Upload Image Via:', ('File', 'URL'))
img = None

if upload_type == 'File':
    uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Invalid file: {e}")
elif upload_type == 'URL':
    url = st.text_input('Enter image URL')
    if url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
        except Exception as e:
            st.error(f"Invalid URL or image: {e}")

if img:
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Find Similar Products'):
        query_feature = get_feature(img)
        if query_feature is None:
            st.error('Could not process uploaded image.')
        else:
            with st.spinner('Searching for matches...'):
                similarities = []
                for i, pf in enumerate(product_features):
                    if pf is not None:
                        sim = F.cosine_similarity(query_feature.unsqueeze(0), pf.unsqueeze(0)).item()
                    else:
                        sim = 0.0
                    similarities.append((sim, i))
                similarities.sort(reverse=True, key=lambda x: x[0])
            
            # Filter by similarity score
            min_score = st.slider('Minimum Similarity Score', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            filtered = [s for s in similarities if s[0] >= min_score]
            
            st.subheader(f'Found {len(filtered)} Similar Products')
            if not filtered:
                st.write('No matches above the threshold. Try lowering the score.')
            for sim, idx in filtered:
                p = products[idx]
                with st.expander(f"{p['title']} (Similarity: {sim:.2f})"):
                    cols = st.columns(2)
                    cols[0].image(p['thumbnail'], width=150)
                    cols[1].write(f"**Name:** {p['title']}")
                    cols[1].write(f"**Category:** {p['category']}")
else:
    st.info('Upload an image to start.')

# Footer
st.markdown('---')
st.caption('Built with Streamlit, PyTorch, and DummyJSON API. Mobile-responsive design.')