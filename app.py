import streamlit as st
import pandas as pd
import pickle
import random
from datetime import datetime, timedelta

# Load recommendation model
with open("models/recommendation_model.pkl", "rb") as f:
    vectorizer, cosine_sim, merged_data = pickle.load(f)

# Function to get recommendations
def recommend_places(location, num_days):
    filtered_data = merged_data[merged_data["location"].str.contains(location, case=False, na=False)]
    
    if filtered_data.empty:
        return None
    
    # Select random destinations from filtered data
    recommended_places = filtered_data.sample(min(num_days, len(filtered_data)))
    
    itinerary = []
    for i, (_, row) in enumerate(recommended_places.iterrows()):
        activities = [
            f"Explore {row['destination_name']}",
            "Try local cuisine",
            "Visit historical sites",
            "Relax at the beach" if "beach" in row["category"].lower() else "Enjoy the nightlife",
            "Capture scenic views",
        ]
        random.shuffle(activities)
        
        itinerary.append({
            "day": i + 1,
            "location": row["destination_name"],
            "staying_at": f"{row['location']} - {row['destination_name']}",
            "description": f"Visit {row['destination_name']} for {row['category'].lower()} experiences.",
            "activities": activities[:4]  # Select 4 random activities
        })
    
    return itinerary

# Streamlit UI
st.title("🌍 AI-Powered Travel Planner")

location = st.text_input("Enter your destination (e.g., Sri Lanka, Paris, Maldives)")
start_date = st.date_input("Select your trip start date", min_value=datetime.today())
end_date = st.date_input("Select your trip end date", min_value=start_date)

if st.button("Generate Itinerary"):
    num_days = (end_date - start_date).days + 1

    if num_days <= 0:
        st.warning("Please select a valid date range.")
    elif location:
        itinerary = recommend_places(location, num_days)

        if itinerary:
            for day in itinerary:
                st.write(f"### **DAY {day['day']}: {day['location']}**")
                st.write(f"**Staying at:** {day['staying_at']}")
                st.write(f"**Description:** {day['description']}")
                st.write("**Recommended Activities:**")
                for activity in day["activities"]:
                    st.write(f"- {activity}")
                st.write("-" * 50)
        else:
            st.error("No recommendations found for this location.")
    else:
        st.warning("Please enter a location!")
