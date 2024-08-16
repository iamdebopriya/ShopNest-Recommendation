import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# Load and cache the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('productsall.csv')  # Assumes _id and orders columns are in the CSV
    return df

# Load the dataset
data = load_data()

# Load pre-trained ResNet model for image classification
resnet_model = ResNet50(weights='imagenet')

# Initialize TF-IDF vectorizer and fit on product descriptions
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['productDesc'])

# Helper function for recommendations
def recommend_products(query, top_n=5):
    query_vector = tfidf_vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return data.iloc[top_indices], similarity_scores[top_indices]

# Helper function to get random products
def get_random_products(n=5):
    return data.sample(n=n)

# Helper function for visual search with uploaded image
def classify_uploaded_image(uploaded_img):
    fashion_terms = ['dress', 'outfit', 'jewelry', 'shoes']
    try:
        # Preprocess the uploaded image
        img = Image.open(uploaded_img).convert('RGB')
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict the class of the image
        preds = resnet_model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=5)[0]  # Get top 5 predictions

        # Extract top class names
        top_class_names = [class_name for (_, class_name, _) in decoded_preds]
        
        # Find the fashion-related terms in predictions
        matched_terms = [term for term in fashion_terms if term in top_class_names]
        return decoded_preds[:3], matched_terms  # Return only top 3 predictions and matched terms
    except Exception as e:
        return str(e), []

# Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #000000; /* Black background */
        color: #ff69b4; /* Pink text color */
        font-family: 'Arial', sans-serif; /* Font style */
    }
    .stTitle {
        color: #ff69b4; /* Pink title color */
        font-family: 'Arial', sans-serif; /* Font style */
        font-size: 1.5em; /* Slightly smaller title size */
        font-weight: bold; /* Title weight */
        margin-bottom: 1.5rem; /* Space below title */
    }
    .stTextArea textarea, .stTextInput input {
        background-color: #000000; /* Black background for textarea and input */
        color: #ff69b4; /* Pink text color for textarea and input */
        border: 1px solid #ff69b4; /* Pink border */
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #ff69b4; /* Pink border on focus */
        box-shadow: 0 0 0 1px #ff69b4; /* Pink outline on focus */
    }
    .stButton > button {
        background-color: #ff69b4; /* Pink button background color */
        color: #000; /* Black button text color */
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #ff1493; /* Darker pink on hover */
    }
    .stImage img {
        width: 150px; /* Fixed image width */
        height: 150px; /* Fixed image height */
        object-fit: cover; /* Ensure images are cropped and fit the dimensions */
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .section {
        padding: 2rem; /* Padding inside each section */
        margin-bottom: 2rem; /* Margin between sections */
        background-color: #1e1e1e; /* Dark background for sections */
        border-radius: 8px; /* Rounded corners for sections */
    }
    .stFileUploader > div {
        background-color: #ff69b4; /* Pink background for file uploader */
        border: 1px solid #ff69b4; /* Pink border */
        color: #000; /* Black text color */
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="stTitle">Fashion Product Catalog</h1>', unsafe_allow_html=True)

# Recommendations Section
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h2 class="stTitle">Get Recommendations by Description and Name</h2>', unsafe_allow_html=True)
user_input = st.text_area("Enter a product description or name:")
if st.button('Get Recommendations'):
    if user_input:
        recommendations, scores = recommend_products(user_input)
        st.markdown('<h3 class="stTitle">Recommended Products</h3>', unsafe_allow_html=True)
        for index, (row, score) in enumerate(zip(recommendations.iterrows(), scores)):
            row = row[1]
            st.write(f"**Product ID:** {row['_id']}")
            st.write(f"**Product Name:** {row['productName']}")
            st.write(f"**Description:** {row['productDesc']}")
            st.write(f"**Price:** ${row['price']:.2f}")
            st.write(f"**Similarity Score:** {score:.2f}")
            if pd.notna(row['productImageURL']):
                response = requests.get(row['productImageURL'])
                img = Image.open(BytesIO(response.content))
                st.image(img, caption=row['productName'], use_column_width=True)
            else:
                st.write("No image available.")
    else:
        st.write("Please enter a description or name to get recommendations.")
st.markdown('</div>', unsafe_allow_html=True)

# Classification Section
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<h2 class="stTitle">Upload Image for Classification and Suggestions</h2>', unsafe_allow_html=True)
uploaded_image = st.file_uploader("Upload an Image for Classification", type=['jpg', 'png', 'jpeg'])
if st.button('Classify Image'):
    if uploaded_image:
        st.markdown('<h3 class="stTitle">Uploaded Image Classification</h3>', unsafe_allow_html=True)
        preds, matched_terms = classify_uploaded_image(uploaded_image)

        if isinstance(preds, str):
            st.write(preds)
        else:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            st.write("Top 3 classifications:")
            for i, (class_id, class_name, score) in enumerate(preds):
                st.write(f"{i + 1}. **{class_name}** ({score:.2f})")

            # Show top 5 fashion-related terms
            st.markdown('<h3 class="stTitle">Top Fashion-Related Terms</h3>', unsafe_allow_html=True)
            if matched_terms:
                for term in matched_terms:
                    st.write(f"**{term.capitalize()}**")
            else:
                st.write("No fashion-related terms matched.")

            # Show random products
            st.markdown('<h3 class="stTitle">Best Product Suggestions</h3>', unsafe_allow_html=True)
            random_products = get_random_products(n=5)
            for index, row in random_products.iterrows():
                st.write(f"**Product ID:** {row['_id']}")
                st.write(f"**Product Name:** {row['productName']}")
                st.write(f"**Description:** {row['productDesc']}")
                st.write(f"**Price:** ${row['price']:.2f}")
                if pd.notna(row['productImageURL']):
                    response = requests.get(row['productImageURL'])
                    img = Image.open(BytesIO(response.content))
                    st.image(img, caption=row['productName'], use_column_width=True)
                else:
                    st.write("No image available.")
    else:
        st.write("Please upload an image to classify.")
st.markdown('</div>', unsafe_allow_html=True)

# Visit our website button at the bottom
st.markdown('<h2 class="stTitle">Visit Our Website</h2>', unsafe_allow_html=True)
st.markdown("""
    <a href="https://shop-nest-olive.vercel.app/" target="_blank">
        <button style="background-color: #ff69b4; color: #000; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">Visit Our Website</button>
    </a>
""", unsafe_allow_html=True)
