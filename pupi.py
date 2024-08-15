import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('productswithid.csv', on_bad_lines='skip', delimiter=',', engine='python')
        return data
    except pd.errors.ParserError as e:
        st.error(f"Error parsing the CSV file: {e}")
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Load data
data = load_data()

# If data is loaded successfully
if data is not None:
    # Combine product name and description into a single column
    data['combined'] = data['productName'] + ' ' + data['productDesc']

    # Create a TF-IDF Vectorizer and fit it to the combined text
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['combined'])

    # Calculate cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Function to recommend products based on the description
    def recommend_products(description, cosine_sim=cosine_sim):
        # Combine user input with a placeholder product name
        input_text = "Query Product " + description
        input_vec = vectorizer.transform([input_text])
        sim_scores = linear_kernel(input_vec, tfidf_matrix).flatten()

        # Get indices of the most similar products
        sim_indices = sim_scores.argsort()[::-1]
        recommendations = []
        for idx in sim_indices:
            if sim_scores[idx] > 0.0:  # Filter out products with a similarity score of 0.0
                recommendations.append({
                    '_id': data.iloc[idx]['_id'],
                    'productName': data.iloc[idx]['productName'],
                    'productDesc': data.iloc[idx]['productDesc'],
                    'price': data.iloc[idx]['price'],
                    'productImageURL': data.iloc[idx]['productImageURL'],
                    'similarity_score': sim_scores[idx]
                })
        return recommendations

    # Streamlit app styling
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
            font-size: 2em; /* Title size */
            font-weight: bold; /* Title weight */
        }
        .stTextArea textarea {
            background-color: #000000; /* Black background for textarea */
            color: #ff69b4; /* Pink text color for textarea */
            border: 1px solid #808080; /* Grey border */
        }
        .stTextArea textarea:focus {
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
        .stTextInput label {
            color: #ff69b4; /* Pink color for label */
        }
        .stImage img {
            width: 150px; /* Fixed image width */
            height: 150px; /* Fixed image height */
            object-fit: cover; /* Ensure images are cropped and fit the dimensions */
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="stTitle">Product Recommendation System</h1>', unsafe_allow_html=True)

    # User input
    user_description = st.text_area("Enter a product description:")

    if st.button('Get Recommendations'):
        if user_description:
            recommendations = recommend_products(user_description)

            if recommendations:
                st.write(f"**Recommended Products:**")
                for rec in recommendations[:5]:  # Show top 5 recommendations
                    st.write(f"**Product ID:** {rec['_id']}")
                    st.write(f"**Product Name:** {rec['productName']}")
                    st.write(f"**Description:** {rec['productDesc']}")
                    st.write(f"**Price:** ${rec['price']}")
                    st.image(rec['productImageURL'])  # Use fixed image size
                    st.write(f"**Similarity Score:** {rec['similarity_score']:.2f}")
                    st.write("---")
                st.write("You can search products with these IDs.")
            else:
                st.write("No recommendations found.")

    # Button to visit the external site
    st.markdown("""
        <a href="https://shop-nest-olive.vercel.app/" target="_blank">
            <button style="background-color: #ff69b4; color: #000; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
                Visit Our Website
            </button>
        </a>
    """, unsafe_allow_html=True)
