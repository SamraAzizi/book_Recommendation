\

# Load the dataset from a CSV file
books = pd.read_csv('books.csv')

# Ensure the CSV has the necessary columns: 'title', 'authors', 'categories', 'description'
if not {'title', 'authors', 'categories', 'description'}.issubset(books.columns):
    raise ValueError("CSV file must contain 'title', 'authors', 'categories', and 'description' columns.")

# Combine features into a single string
books['combined_features'] = books['categories'] + ' ' + books['authors'] + ' ' + books['description']

# Create a TF-IDF Vectorizer to convert combined features to vectors
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the data
tfidf_matrix = tfidf.fit_transform(books['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get book recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the book that matches the title
    try:
        idx = books.index[books['title'] == title][0]
    except IndexError:
        return "Book not found in the dataset."

    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 3 most similar books
    sim_scores = sim_scores[1:4]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 3 most similar books
    return books['title'].iloc[book_indices].tolist()

# Main function to interact with the user
if __name__ == "__main__":
    user_input = input("Enter a book title to get recommendations: ")
    recommendations = get_recommendations(user_input)
    if isinstance(recommendations, list):
        print("Recommended books:")
        for book in recommendations:
            print(f"- {book}")
    else:
        print(recommendations)
