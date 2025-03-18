import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
destinations = pd.read_csv("data/Expanded_Destinations.csv")
reviews = pd.read_csv("data/Final_Updated_Expanded_Reviews.csv")
user_history = pd.read_csv("data/Final_Updated_Expanded_UserHistory.csv")
users = pd.read_csv("data/Final_Updated_Expanded_Users.csv")

# Merge user history with user data
user_data = pd.merge(user_history, users, on="user_id", how="left")

# Merge with destinations and reviews
merged_data = pd.merge(user_data, destinations, on="destination_id", how="left")
merged_data = pd.merge(merged_data, reviews, on="destination_id", how="left")

# Feature Engineering: Combine relevant text features
merged_data["combined_text"] = merged_data["destination_name"] + " " + merged_data["category"] + " " + merged_data["review_text"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(merged_data["combined_text"])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save model
with open("models/recommendation_model.pkl", "wb") as f:
    pickle.dump((vectorizer, cosine_sim, merged_data), f)

print("Model saved successfully!")