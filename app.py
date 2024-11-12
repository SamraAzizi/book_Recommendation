 import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Set page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# Add a title and description
st.title("📚 Book Recommendation System")
st.write("Enter a book title and get personalized book recommendations!")

# Load and process the data
@st.cache_data  # This decorator caches the data
def load_data():
    books = pd.read_csv('books.csv')
    return books

def get_recommendations(title, books, cosine_sim):
    # Get the index of the book that matches the title
    if title not in books['title'].values:
        return "Book not found in the dataset."

    idx = books.index[books['title'] == title][0]

    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar books (excluding the book itself)
    sim_scores = sim_scores[1:6]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar books with their details
    recommended_books = books.iloc[book_indices][['title', 'authors', 'categories', 'average_rating']]
    return recommended_books

try:
    # Load the data
    books = load_data()

    # Create combined features
    books['combined_features'] = books['categories'] + ' ' + books['authors'] + ' ' + books['description']

    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books['combined_features'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Create a dropdown with all book titles
    book_titles = sorted(books['title'].tolist())
    selected_title = st.selectbox(
        "Select a book you like:",
        book_titles
    )

    if st.button('Get Recommendations'):
        # Get recommendations
        recommendations = get_recommendations(selected_title, books, cosine_sim)
        
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.success("Here are your recommended books:")
            
            # Display recommendations in a nice format
            for idx, row in recommendations.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(f"**{row['title']}**")
                    with col2:
                        st.write(f"by {row['authors']}")
                    with col3:
                        st.write(f"Rating: {row['average_rating']:.2f}")
                    st.write(f"Categories: {row['categories']}")
                    st.divider()

    # Add some additional information
    with st.sidebar:
        st.header("About")
        st.write("""
        This book recommendation system uses content-based filtering to suggest books 
        similar to the one you select. It considers factors such as:
        - Book categories
        - Authors
        - Book descriptions
        """)
        
        st.header("Statistics")
        st.write(f"Total books in database: {len(books)}")
        st.write(f"Number of unique authors: {books['authors'].nunique()}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please ensure that the 'books.csv' file is in the correct location and format.")