import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template, abort
import os
import time
current_time = time.time()  # Get the current time
app = Flask(__name__) # Create a Flask web server

# Global variable to hold the preprocessed data
data = None

# Function to load the data
def load_data(filepath):
    # return the data as a pandas dataframe
    return pd.read_csv(filepath)
    

# Function to preprocess the data
# Considering that the genres are of the form A, B, C, D and we need to get each and every one so that we know what
# genres an anime belongs to. This is possible if we split it into a list. We split the genres by looking for the commas(,).
# I could've used an anonymous function but I don't remember them properly enough to try them when I can do it like this.
# Also checks in case genres are not already in the form of a list and then splits the string into a list, if it is a list 
# it returns it.

def preprocess_data(anime_list_data):
    anime_list_data = anime_list_data.dropna(subset=['genres', 'score'])
    def genre_split(genre_string):
        if not isinstance(genre_string, list):
            return genre_string.split(', ')
        return genre_string
    anime_list_data['genres'] = anime_list_data['genres'].apply(genre_split)
    return anime_list_data

print(os.getcwd())
# Load the preprocessing objects from disk
mlb_model_one = joblib.load('models/mlb_model_one.pkl')
scaler_model_one = joblib.load('models/scaler_model_one.pkl')

mlb_model_two = joblib.load('models/mlb_model_two.pkl')
scaler_score_model_two = joblib.load('models/scaler_score_model_two.pkl')
scaler_year_model_two = joblib.load('models/scaler_year_model_two.pkl')

mlb_model_three = joblib.load('models/mlb_model_three.pkl')
scaler_model_three = joblib.load('models/scaler_model_three.pkl')

mlb_model_four = joblib.load('models/mlb_model_four.pkl')
scaler_score_model_four = joblib.load('models/scaler_score_model_four.pkl')
scaler_year_model_four = joblib.load('models/scaler_year_model_four.pkl')

# Function to extract the release year from the 'aired' column
def extract_year(aired_string):
        match = re.search(r'\d{4}', aired_string)
        return int(match.group()) if match else None

# Function to get recommendations
def get_recommendations_model_one(title, anime_list_data, model, n):
    # Get the index of the anime
    idx = anime_list_data[anime_list_data['title'] == title].index[0]

    # Preprocess the genres and scores
    encoded_genres = mlb_model_one.transform(anime_list_data['genres'])
    scores_normalized = scaler_model_one.transform(anime_list_data[['score']])

    # Concatenate preprocessed features
    features = np.concatenate([encoded_genres, scores_normalized], axis=1)

    # Generate embeddings for all animes using the trained model
    anime_embeddings = model.predict(features)

    # Compute cosine similarity
    cos_sim = cosine_similarity([anime_embeddings[idx]], anime_embeddings)

    # Get top n most similar animes
    top_indices = np.argsort(cos_sim[0])[-n-1:-1][::-1]
    return anime_list_data['title'].iloc[top_indices].tolist()


def get_recommendations_time(title, anime_list_data, model, n):
    # Get the index of the anime
    idx = anime_list_data[anime_list_data['title'] == title].index[0]

    # Preprocess the genres, scores, and release years
    encoded_genres = mlb_model_two.transform(anime_list_data['genres'])
    scores_normalized = scaler_score_model_two.transform(anime_list_data[['score']])
    release_years = anime_list_data['aired'].apply(extract_year).fillna(0).to_numpy().reshape(-1, 1)
    #.reshape(-1, 1) is used to reshape the array to a single column array
    release_years_normalized = scaler_year_model_two.transform(release_years)

    # Concatenate preprocessed features
    features = np.concatenate([encoded_genres, scores_normalized, release_years_normalized], axis=1)

    # Generate embeddings for all animes using the trained model
    anime_embeddings_time_based = model.predict(features)

    # Compute cosine similarity
    cos_sim = cosine_similarity([anime_embeddings_time_based[idx]], anime_embeddings_time_based)

    # Get top n most similar animes
    top_indices = np.argsort(cos_sim[0])[-n-1:-1][::-1]
    return anime_list_data['title'].iloc[top_indices].tolist()


def get_recommendations_explainable(anime_title, anime_list_data, model, n):
    # Get the index of the anime in the dataframe
    idx = anime_list_data[anime_list_data['title'] == anime_title].index[0]

    # Preprocess the genres and scores
    encoded_genres = mlb_model_three.transform(anime_list_data['genres'])
    scores_normalized = scaler_model_three.transform(anime_list_data[['score']])

    # Concatenate preprocessed features
    features = np.concatenate([encoded_genres, scores_normalized], axis=1)

    # Generate embeddings for all animes using the trained model
    anime_embeddings = model.predict(features)

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(anime_embeddings[idx].reshape(1, -1), anime_embeddings)

    # Get the indices of the anime sorted by cosine similarity (in descending order)
    sorted_indices = np.argsort(cosine_similarities[0])[::-1]

    # Get the titles of the top N anime
    recommended_anime = anime_list_data.iloc[sorted_indices[1:n+1]]

    # Get the explanations for the recommendations
    recommendations = []

    # Iterate over the recommended anime
    for i in recommended_anime.index:
        # Get the common genres between the recommended anime and the anime we are explaining
        common_genres = set(anime_list_data.iloc[idx]['genres']).intersection(set(anime_list_data.iloc[i]['genres']))
        
        # Create the explanation
        explanation = f"Recommended because of similar genres: {', '.join(common_genres)} and similar scores: {recommended_anime.loc[i, 'score']:.2f} vs {anime_list_data.iloc[idx]['score']:.2f}"
        
        # Add the explanation to the list
        recommendations.append((recommended_anime.loc[i, 'title'], explanation))

    return recommendations

# Function to get recommendations with narrative explanations and time-based embeddings
def get_recommendations_with_narrative_explanation_and_time(title, anime_list_data, model, n):
    # Get the index of the anime
    idx = anime_list_data[anime_list_data['title'] == title].index[0]

    # Preprocess the genres, scores, and release years
    encoded_genres = mlb_model_four.transform(anime_list_data['genres'])
    scores_normalized = scaler_score_model_four.transform(anime_list_data[['score']])
    release_years = anime_list_data['aired'].apply(extract_year).fillna(0).to_numpy().reshape(-1, 1)
    release_years_normalized = scaler_year_model_four.transform(release_years)

    # Concatenate preprocessed features
    features = np.concatenate([encoded_genres, scores_normalized, release_years_normalized], axis=1)

    # Generate embeddings for all animes using the trained model
    anime_embeddings_explanation_time = model.predict(features)

    # Compute cosine similarity
    cos_sim = cosine_similarity([anime_embeddings_explanation_time[idx]], anime_embeddings_explanation_time)

    # Get top n most similar animes
    top_indices = np.argsort(cos_sim[0])[-n-1:-1][::-1]

    # Get the explanations for the recommendations
    recommendations = []
    for i in top_indices:
        # Get the title of the recommended anime
        anime_title = anime_list_data.iloc[i]['title']
        
        # Get the common genres between the recommended anime and the anime we are explaining
        common_genres = set(anime_list_data.iloc[idx]['genres']).intersection(set(anime_list_data.iloc[i]['genres']))
        
        # Get the score of the current anime
        score_current = anime_list_data.iloc[idx]['score']
        
        # Get the score of the recommended anime
        score_recommended = anime_list_data.iloc[i]['score']

        # Create the explanation
        explanation = f"Recommended because of similar genres: {', '.join(common_genres)} and similar scores: {score_recommended:.2f} vs {score_current:.2f}"
        
        # Append a tuple with the title and explanation
        recommendations.append((anime_title, explanation))
    return recommendations


# Load the trained models from disk
model_one = tf.keras.models.load_model('models/model_one')
model_two = tf.keras.models.load_model('models/model_two')
model_three = tf.keras.models.load_model('models/model_three')
model_four = tf.keras.models.load_model('models/model_four')


# Function to make predictions with each model
def make_predictions(title, n, data):
    # Make predictions with each model
    recommendations_model1 = get_recommendations_model_one(title, data, model_one, n)
    recommendations_model2 = get_recommendations_time(title, data, model_two, n)
    recommendations_model3 = get_recommendations_explainable(title, data, model_three, n)
    recommendations_model4 = get_recommendations_with_narrative_explanation_and_time(title, data, model_four, n)

    # Return the recommendations
    recommendations = {
        'model1': recommendations_model1,
        'model2': recommendations_model2,
        'model3': recommendations_model3,
        'model4': recommendations_model4
    }
    return recommendations

@app.route('/anime-titles') # Route to get the anime titles
def anime_titles():
    global data # Use the global variable
    if data is None: # If data is not loaded
        filepath = 'rec_anime_list.csv' # Path to the data
        data = load_data(filepath) # Load the data
        data = preprocess_data(data) # Preprocess the data
    titles = data['title'].unique().tolist() # Get the unique anime titles
    return jsonify({"titles": titles}) # Return the titles as JSON

# Route to handle the index page
@app.route('/')
def index(): 
    return render_template('index.html') # Render the index.html template

# Route to handle the recommendations
@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    try:
        # Debug: print raw request data and JSON payload
        print("Raw request data:", request.data)
        print("JSON payload:", request.get_json()) 

        content = request.get_json() # Get the JSON payload
        title = content['title'] # Get the title from the payload
        n = int(content.get('n', 5)) # Get the number of recommendations from the payload

        global data
        if data is None:  # If data is not loaded
            filepath = 'rec_anime_list.csv'  # Path to the data
            data = load_data(filepath)  # Load the data
            data = preprocess_data(data)  # Preprocess the data

        predictions = make_predictions(title, n, data)  # Make predictions
        
        # Debug: print the predictions being sent back to the client
        print("Predictions:", predictions)
        current_time2 = time.time()  # Get the current time
        difference_time = current_time2 - current_time  # Get the time taken to make the predictions
        print("Time taken:", difference_time)  # Print the time taken to make the predictions
        return jsonify(predictions)  # Return the predictions as JSON
    except Exception as e:
        print(e)  # Print the error to the console
        return jsonify({"error": str(e)}), 400  # Return a HTTP status code of 400




if __name__ == '__main__':
    
    app.run(debug=True)  # This starts the Flask web server