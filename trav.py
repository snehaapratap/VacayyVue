import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
destinations = pd.read_csv("data/Expanded_Destinations.csv")
reviews = pd.read_csv("data/Final_Updated_Expanded_Reviews.csv")
user_history = pd.read_csv("data/Final_Updated_Expanded_UserHistory.csv")
users = pd.read_csv("data/Final_Updated_Expanded_Users.csv")

# Ensure column names are correct
print("Destinations Columns:", destinations.columns)
print("Reviews Columns:", reviews.columns)
print("User History Columns:", user_history.columns)
print("Users Columns:", users.columns)

# Merge user history with users
user_data = pd.merge(user_history, users, on="UserID", how="left")

# Merge with destinations and reviews
merged_data = pd.merge(user_data, destinations, on="DestinationID", how="left")
merged_data = pd.merge(merged_data, reviews, on="DestinationID", how="left")

# Create a new column with relevant text information
merged_data["combined_text"] = (
    merged_data["Name"] + " " +
    merged_data["Type"] + " " +
    merged_data["ReviewText"]
)
# Convert text data to numerical representation using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(merged_data["combined_text"].fillna(""))

# Compute similarity scores
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save model
with open("models/recommendation_model.pkl", "wb") as f:
    pickle.dump((vectorizer, cosine_sim, merged_data), f)

print("Model saved successfully!")

def recommend_places(user_input):
    input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vector, cosine_sim).flatten()
    indices = similarities.argsort()[-5:][::-1]
    return merged_data.iloc[indices][["destination_name", "category", "rating"]]

print(recommend_places("beach adventure"))
