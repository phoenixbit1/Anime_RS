# Anime Recommendation System using Neural Networks

## Description
This repository contains a neural network-based anime recommendation system that evaluates user preferences across four models:
1. A model using genre and score.
2. A model incorporating genre, score, and release time.
3. A model providing explanations for recommendations along with genre and score.
4. An integrated model combining genre, score, release time, and explanation.

Our hypothesis posits that the fourth model will yield the most satisfactory recommendations.

### Dependencies
Ensure you have the following prerequisites installed on your machine:
- tensorflow 2.13.0
- numpy 1.24.3
- pandas 2.0.2
- Flask 2.3.2
- scikit-learn 1.4.0
- joblib 1.3.2
- keras 2.13.1
- python 3.11.0

These can be installed via `requirements.txt` using the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Running the Code

To train the models, execute the training script by running the following command:

```bash
python Recommendation_model_train.py
```

This script will conduct the training process for all four models and save the trained models to the disk.

After the models have been trained, you can initiate the web application for making predictions with:

```bash
python Recommendation_model_predict.py
```

Launching this script will start the Flask web server. You can then navigate to the local URL provided by the Flask output to interact with the web application. 





