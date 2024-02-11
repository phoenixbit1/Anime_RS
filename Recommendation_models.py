import logging
import sys
import json
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging to print to stderr for warnings and above
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


# Function to load the data
def load_data(filepath):
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


# pd.get_dummies() from pandas and OneHotEncoder from skLearn.preprocessing are designed for single label scenarios.
# MultiLabelBinarizer is designed for multi-label scenarios. It takes a list of lists as input and returns a matrix.

# Function to train the first neural network model
def model_one(anime_list_data, embedding_size, epochs, batch_size):
    # One-hot encode the genres
    mlb = MultiLabelBinarizer()
    encoded_genres = mlb.fit_transform(anime_list_data['genres'])
    
    # Normalize the scores
    scaler = MinMaxScaler()
    scores_normalized = scaler.fit_transform(anime_list_data[['score']])
    
    # Concatenate the encoded genres and normalized scores to create the features array
    features = np.concatenate([encoded_genres, scores_normalized], axis=1)
    
    # Define the encoder
    input_layer = Input(shape=(features.shape[1],))
    encoder_layer = Dense(embedding_size, activation='relu')(input_layer)

    # Define the decoder
    decoder_layer = Dense(features.shape[1], activation='sigmoid')(encoder_layer)

    # Define the autoencoder model
    autoencoder = Model(input_layer, decoder_layer)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    autoencoder.fit(features, features, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Define and return the encoder part of the autoencoder for generating embeddings
    encoder = Model(input_layer, encoder_layer)
    anime_embeddings = encoder.predict(features)
    return anime_embeddings

# Function to extract the release year from the 'aired' column
def extract_year(aired_string):
        match = re.search(r'\d{4}', aired_string)
        return int(match.group()) if match else None

# Function to train the second neural network model
def model_two(anime_list_data, embedding_size, epochs, batch_size):
    # Extract the release year from the 'aired' column
    anime_list_data['release_year'] = anime_list_data['aired'].apply(extract_year)

    # Drop rows with missing 'release_year'
    anime_list_data = anime_list_data.dropna(subset=['release_year'])

    # One-hot encode the genres
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(anime_list_data['genres'])

    # Normalize the scores
    scaler_score = MinMaxScaler()
    scores_normalized = scaler_score.fit_transform(anime_list_data[['score']])

    # Normalize the release year
    scaler_year = MinMaxScaler()
    release_year_normalized = scaler_year.fit_transform(anime_list_data[['release_year']])

    # Concatenate the encoded genres, normalized scores, and normalized release year to create the features array
    anime_features_time_based = np.concatenate([genres_encoded, scores_normalized, release_year_normalized], axis=1)

    # Define the encoder
    input_layer = Input(shape=(anime_features_time_based.shape[1],))
    encoder_layer = Dense(embedding_size, activation='relu')(input_layer)

    # Define the decoder
    decoder_layer = Dense(anime_features_time_based.shape[1], activation='sigmoid')(encoder_layer)

    # Define the autoencoder model
    autoencoder_time_based = Model(input_layer, decoder_layer)

    # Compile the model
    autoencoder_time_based.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    autoencoder_time_based.fit(anime_features_time_based, anime_features_time_based, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Define the encoder model
    encoder_time_based = Model(input_layer, encoder_layer)

    # Compute the embeddings for all animes
    anime_embeddings_time_based = encoder_time_based.predict(anime_features_time_based)
    return anime_embeddings_time_based

# Function to train the third neural network model
def model_three(anime_list_data, embedding_size ,epochs, batch_size):
    # Preprocess the data
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(anime_list_data['genres'])
    
    # Normalize the scores
    scaler = MinMaxScaler()
    scores_normalized = scaler.fit_transform(anime_list_data[['score']])
    
    # Concatenate the encoded genres and normalized scores to create the features array
    anime_features = np.concatenate([genres_encoded, scores_normalized], axis=1)

    # Define the encoder
    input_layer = Input(shape=(anime_features.shape[1],))
    encoder_layer = Dense(embedding_size, activation='relu')(input_layer)

    # Define the decoder
    decoder_layer = Dense(anime_features.shape[1], activation='sigmoid')(encoder_layer)

    # Define the autoencoder model
    autoencoder = Model(input_layer, decoder_layer)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    autoencoder.fit(anime_features, anime_features, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Define and return the encoder part of the autoencoder
    encoder = Model(input_layer, encoder_layer)
    anime_embeddings_explain = encoder.predict(anime_features)
    return anime_embeddings_explain

# Function to train the fourth neural network model
def model_four(anime_list_data, embedding_size ,epochs, batch_size):
    # Extract the release year from the 'aired' column
    anime_list_data['release_year'] = anime_list_data['aired'].apply(extract_year)

    # Drop rows with missing 'release_year'
    anime_list_data.dropna(subset=['release_year'], inplace=True)

    # One-hot encode genres
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(anime_list_data['genres'])

    # Normalize scores and release years
    scaler = MinMaxScaler()
    scores_normalized = scaler.fit_transform(anime_list_data[['score']])
    release_year_normalized = scaler.fit_transform(anime_list_data[['release_year']])

    # Concatenate the encoded genres, normalized scores, and normalized release years to create the features array
    features = np.concatenate([genres_encoded, scores_normalized, release_year_normalized], axis=1)

    # Define the encoder
    input_layer = Input(shape=(features.shape[1],))
    encoder_layer = Dense(embedding_size, activation='relu')(input_layer)

    # Define the decoder
    decoder_layer = Dense(features.shape[1], activation='sigmoid')(encoder_layer)

    # Define the autoencoder model
    autoencoder = Model(input_layer, decoder_layer)
    
    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Model Training
    autoencoder.fit(features, features, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Encoder Model for Embeddings
    encoder = Model(input_layer, encoder_layer)
    anime_embeddings_best = encoder.predict(features)
    return anime_embeddings_best

# Function to get recommendations using the first model
def get_recommendations(title, anime_list_data, anime_embeddings, n):
    # Get the index of the anime
    idx = anime_list_data[anime_list_data['title'] == title].index[0]

    # Compute the cosine similarity between the anime's embedding and all other embeddings
    cos_sim = cosine_similarity([anime_embeddings[idx]], anime_embeddings)

    # Get the indices of the top n most similar animes
    top_indices = np.argsort(cos_sim[0])[-n-1:-1][::-1]

    # Return their titles
    return anime_list_data['title'].iloc[top_indices].tolist()

# Function to get recommendations using the second model
def get_recommendations_time(title, anime_list_data, anime_embeddings_time_based, n):
    # Get the index of the anime
    idx = anime_list_data[anime_list_data['title'] == title].index[0]

    # Compute the cosine similarity between the anime's embedding and all other embeddings
    cos_sim = cosine_similarity([anime_embeddings_time_based[idx]], anime_embeddings_time_based)

    # Get the indices of the top n most similar animes
    top_indices = np.argsort(cos_sim[0])[-n-1:-1][::-1]

    # Return their titles
    return anime_list_data['title'].iloc[top_indices].tolist()

# Function to get recommendations using the third model
def get_recommendations_explainable(anime_title,  anime_list_data, anime_embeddings,n):
    # Get the index of the anime in the dataframe
    anime_idx = anime_list_data[anime_list_data['title'] == anime_title].index[0]

    # Get the embedding of the anime
    anime_embedding = anime_embeddings[anime_idx]

    # Compute the cosine similarity between the anime and all other anime
    cosine_similarities = cosine_similarity(anime_embedding.reshape(1, -1), anime_embeddings)

    # Get the indices of the anime sorted by cosine similarity (in descending order)
    sorted_indices = np.argsort(cosine_similarities[0])[::-1]

    # Get the titles of the top N anime
    recommended_anime = anime_list_data.iloc[sorted_indices[1:n+1]]

    # Generate explanations for the recommendations
    explanations = []
    for anime_idx, row in recommended_anime.iterrows(): # Iterate over the recommended anime
        common_genres = set(anime_list_data.iloc[anime_idx]['genres']).intersection(set(row['genres'])) # Find the common genres
        explanation = f"Recommended because of similar genres: {', '.join(common_genres)} and similar scores: {row['score']:.2f} vs {anime_list_data.iloc[anime_idx]['score']:.2f}" # Generate the explanation
        explanations.append((row['title'], explanation)) # Add the explanation to the list

    return explanations 

# Function to get recommendations using the fourth model
def get_recommendations_with_narrative_explanation_and_time(title, anime_list_data, anime_embeddings_explanation_time, n):
    
    # Get the index of the anime
    idx = anime_list_data[anime_list_data['title'] == title].index[0]

    # Compute the cosine similarity between the anime's embedding and all other embeddings
    cos_sim = cosine_similarity([anime_embeddings_explanation_time[idx]], anime_embeddings_explanation_time)

    # Get the indices of the top n most similar animes
    top_indices = np.argsort(cos_sim[0])[-n-1:-1][::-1]

    # Generate explanations for the recommendations
    recommendations = []

    # Iterate over the top n most similar animes
    for i in top_indices:
        # Get the title of the anime
        anime_title = anime_list_data.iloc[i]['title']

        # Find the common genres
        common_genres = set(anime_list_data.iloc[idx]['genres']).intersection(set(anime_list_data.iloc[i]['genres']))
        
        # get the scores of the anime
        score_current = anime_list_data.iloc[idx]['score']
        
        # get the scores of the recommended anime
        score_recommended = anime_list_data.iloc[i]['score']

        # Generate the explanation
        explanation = f"{anime_title}: Recommended because of similar genres: {', '.join(common_genres)} and similar scores: {score_recommended:.2f} vs {score_current:.2f}"
        recommendations.append(explanation) # Add the explanation to the list
    return recommendations


# Main function 
def main(anime_title):
    try:
        # Load and preprocess data
        filepath = './rec_anime_list.csv'
        data = load_data(filepath)
        data = preprocess_data(data)

        # Hyperparameters
        embedding_size=50
        epochs=35
        batch_size=128
        n=5
        # Train and use the model
        model1_output = model_one(data, embedding_size, epochs, batch_size)
        recommendations_model1 = get_recommendations(anime_title, data, model1_output, n)

        model2_output = model_two(data, embedding_size, epochs, batch_size)
        recommendations_model2 = get_recommendations_time(anime_title, data, model2_output, n)

        model3_output = model_three(data, embedding_size, epochs, batch_size)
        recommendations_model3 = get_recommendations_explainable(anime_title, data, model3_output,n)

        model4_output = model_four(data, embedding_size, epochs, batch_size)
        recommendations_model4 = get_recommendations_with_narrative_explanation_and_time(anime_title, data, model4_output, n)

        # Format recommendations for display on frontend
        recommendations = {
            'model1': recommendations_model1,
            'model2': recommendations_model2,
            'model3': recommendations_model3,
            'model4': recommendations_model4
        }

        # Print out the JSON string
        print(json.dumps(recommendations))  

    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) > 1: # If the user has provided an anime title as a command line argument
        user_input = sys.argv[1] # Use the provided anime title
    else:
        user_input = "Naruto" # Example anime title
    main(user_input)  # Call the main function with the user input
