

## Features

- **Personalized Recommendations**: Get book recommendations based on your reading preferences and previously enjoyed books.
- **User -Friendly Interface**: An intuitive and visually appealing interface built with Streamlit.
- **Data Visualization**: Visualize book statistics and ratings using Plotly.
- **AI Integration**: Utilizes the Ollama LLM for generating recommendations based on user input.

## Technologies Used

- **Python**: The programming language used for the application.
- **Streamlit**: A framework for building web applications for machine learning and data science projects.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning algorithms and data processing.
- **Plotly**: For creating interactive visualizations.
- **Langchain**: For integrating language models and generating recommendations.

## Installation

To run this application, you need to have Python installed on your machine. Follow these steps to set up the project:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/book-recommendation-system.git
   cd book-recommendation-system
   ```
   # Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

   # Data

The application uses a CSV file named `books.csv`, which should contain the following columns:

- `title`: The title of the book.
- `authors`: The authors of the book.
- `categories`: The genres or categories of the book.
- `description`: A brief description of the book.
- `average_rating`: The average rating of the book.

Make sure to format the CSV file correctly to ensure the application runs smoothly.
