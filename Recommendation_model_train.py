import logging
import joblib
import os
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


logging.basicConfig(level=logging.INFO) # Set logging level to INFO

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
    try:
        # One-hot encode the genres
        mlb = MultiLabelBinarizer() 
        encoded_genres = mlb.fit_transform(anime_list_data['genres'])
        joblib.dump(mlb, 'Anime_RS/models/mlb_model_one.pkl')  # Save MultiLabelBinarizer
        
        # MinMaxScaler is used to scale the scores between 0 and 1
        scaler = MinMaxScaler()
        scores_normalized = scaler.fit_transform(anime_list_data[['score']])
        joblib.dump(scaler, 'Anime_RS/models/scaler_model_one.pkl')  # Save MinMaxScaler
        
        features = np.concatenate([encoded_genres, scores_normalized], axis=1)

        # Input layer is the first layer of the neural network
        input_layer = Input(shape=(features.shape[1],))

        # Encoder layer is the layer that is used to encode the data and is the second layer of the neural network
        encoder_layer = Dense(embedding_size, activation='relu')(input_layer)

        # The decoder layer is the layer that is used to decode the data and is the third layer of the neural network
        decoder_layer = Dense(features.shape[1], activation='sigmoid')(encoder_layer)

        # The autoencoder is the neural network that is used to encode and decode the data
        autoencoder = Model(input_layer, decoder_layer)

        # Compile the autoencoder
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(features, features, epochs=epochs, batch_size=batch_size, validation_split=0.2)

        # Save the encoder part of the model
        encoder = Model(input_layer, encoder_layer)
        model_path = 'Anime_RS/models/model_one'
        encoder.save(model_path)
        logging.info(f"Saved model to {model_path}")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Function to extract the release year from the 'aired' column
def extract_year(aired_string):
        match = re.search(r'\d{4}', aired_string)
        return int(match.group()) if match else None

# Function to train the second neural network model
def model_two(anime_list_data, embedding_size, epochs, batch_size):
    try:
        # Extract the release year from the 'aired' column
        anime_list_data['release_year'] = anime_list_data['aired'].apply(extract_year)
        
        # Drop rows with missing 'release_year'
        anime_list_data = anime_list_data.dropna(subset=['release_year'])

        # One-hot encode the genres
        mlb = MultiLabelBinarizer()
        genres_encoded = mlb.fit_transform(anime_list_data['genres'])
        # Save MultiLabelBinarizer object
        joblib.dump(mlb, 'Anime_RS/models/mlb_model_two.pkl')
        
        # Normalize the scores
        scaler_score = MinMaxScaler()
        scores_normalized = scaler_score.fit_transform(anime_list_data[['score']])
        # Save MinMaxScaler for scores
        joblib.dump(scaler_score, 'Anime_RS/models/scaler_score_model_two.pkl')
        
        # Normalize the release year
        scaler_year = MinMaxScaler()
        release_year_normalized = scaler_year.fit_transform(anime_list_data[['release_year']])
        # Save MinMaxScaler for release year
        joblib.dump(scaler_year, 'Anime_RS/models/scaler_year_model_two.pkl')
        
        # Concatenate the encoded genres, normalized scores, and normalized release year to create the features array
        features = np.concatenate([genres_encoded, scores_normalized, release_year_normalized], axis=1)

        # Neural network building and training steps
        input_layer = Input(shape=(features.shape[1],))

        # Encoder layer is the layer that is used to encode the data and is the second layer of the neural network
        encoder_layer = Dense(embedding_size, activation='relu')(input_layer)

        # The decoder layer is the layer that is used to decode the data and is the third layer of the neural network
        decoder_layer = Dense(features.shape[1], activation='sigmoid')(encoder_layer)

        # The autoencoder is the neural network that is used to encode and decode the data
        autoencoder = Model(input_layer, decoder_layer)

        # Compile the autoencoder
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(features, features, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        
        # Save the encoder part of the model
        encoder = Model(input_layer, encoder_layer)
        model_path = 'Anime_RS/models/model_two'
        encoder.save(model_path)
        logging.info(f"Saved model to {model_path}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")




# Training and saving the models
if __name__ == '__main__':
    try:
        # Load the data
        # logging.info("Script started") # Log that the script has started
        filepath = 'Anime_RS/rec_anime_list.csv'
        # logging.info(f"Loading data from {filepath}") # Log that the data is being loaded
        data = load_data(filepath)
        # logging.info("Data loaded") # Log that the data has been loaded
        # logging.info("Preprocessing data") # Log that the data is being preprocessed
        data = preprocess_data(data)
        # logging.info("Data preprocessed") # Log that the data has been preprocessed


        # Set the hyperparameters
        embedding_size = 50 
        epochs = 35
        batch_size = 128


        # logging.info("Training models")
        # make a directory for the models if it doesn't exist
        os.makedirs('Anime_RS/models', exist_ok=True)

        
        # Train each of the models
        logging.info("Training model 1")
        model1_output = model_one(data, embedding_size, epochs, batch_size)

        logging.info("Training model 2")
        model2_output = model_two(data, embedding_size, epochs, batch_size)

        
    except Exception as e:
        logging.exception("An exception occurred: {}".format(e))
