import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .book-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sidebar-content {
        padding: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Add a title and description with custom styling
st.markdown("""
    <h1 style='text-align: center; color: #ff4b4b;'>📚 AI-Powered Book Recommendation System</h1>
    <p style='text-align: center; font-size: 1.2em;'>Get personalized book recommendations using AI!</p>
    <hr>
""", unsafe_allow_html=True)

# Initialize Ollama
@st.cache_resource
def init_ollama():
    return Ollama(model="llama2")

# Load and process the data
@st.cache_data
def load_data():
    books = pd.read_csv('books.csv')
    books['categories'] = books['categories'].fillna('')
    books['authors'] = books['authors'].fillna('')
    books['description'] = books['description'].fillna('')
    books['published_year'] = pd.to_numeric(books['published_year'], errors='coerce')
    return books

def get_ai_recommendations(book_info, user_preferences, llm):
    prompt = PromptTemplate(
        input_variables=["book_info", "user_preferences"],
        template="""
        Based on the following book information:
        {book_info}
        
        And user preferences:
        {user_preferences}
        
        Recommend 5 similar books from the dataset. For each recommendation, provide:
        1. Title
        2. Author
        3. Brief explanation of why it's recommended
        
        Format the response as a structured list with clear separations between recommendations.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(book_info=book_info, user_preferences=user_preferences)
    return response

try:
    # Load the data and initialize Ollama
    books = load_data()
    llm = init_ollama()

    # Sidebar content
    with st.sidebar:
        st.markdown("""
            <div class='sidebar-content'>
                <h2>🤖 AI Preferences</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # User preferences
        genre_preference = st.multiselect(
            "Preferred Genres",
            options=sorted(set([genre.strip() for genres in books['categories'].unique() for genre in genres.split(',') if genre.strip()])),
            help="Select one or more genres you enjoy"
        )
        
        reading_level = st.select_slider(
            "Reading Level",
            options=['Easy', 'Medium', 'Advanced'],
            value='Medium'
        )
        
        mood = st.select_slider(
            "Book Mood",
            options=['Light & Fun', 'Neutral', 'Dark & Serious'],
            value='Neutral'
        )
        
        # Show statistics
        st.markdown("### 📈 Statistics")
        st.write(f"Total Books: {len(books):,}")
        st.write(f"Unique Authors: {books['authors'].nunique():,}")
        avg_rating = books['average_rating'].mean()
        st.write(f"Average Rating: {avg_rating:.2f}⭐")

        # Add a rating distribution plot
        fig = px.histogram(books, x='average_rating', nbins=20,
                          title='Rating Distribution',
                          labels={'average_rating': 'Rating', 'count': 'Number of Books'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_title = st.selectbox(
            "🔍 Select a book you enjoyed:",
            options=sorted(books['title'].unique()),
            help="Type or select a book title"
        )

    with col2:
        if st.button('Get AI Recommendations 🤖'):
            st.session_state.show_recommendations = True

    # Show recommendations
    if st.session_state.get('show_recommendations', False):
        # Get selected book info
        selected_book = books[books['title'] == selected_title].iloc[0]
        book_info = f"""
        Title: {selected_book['title']}
        Author: {selected_book['authors']}
        Categories: {selected_book['categories']}
        Description: {selected_book['description']}
        """
        
        # Prepare user preferences
        user_preferences = f"""
        Preferred Genres: {', '.join(genre_preference)}
        Reading Level: {reading_level}
        Preferred Mood: {mood}
        """
        
        with st.spinner('🤖 AI is generating personalized recommendations...'):
            recommendations = get_ai_recommendations(book_info, user_preferences, llm)
            
            st.success("🎯 Here are your AI-powered recommendations:")
            st.markdown(recommendations)

    # Add footer
    st.markdown("""
        <hr>
        <p style='text-align: center; color: #888;'>
            Powered by Ollama LLM | Made with ❤️
        </p>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please ensure that all requirements are installed and Ollama is running.") 