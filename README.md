# Urdu-Sentiment-Analysis

This project focuses on implementing sequence-based deep learning models for Urdu sentiment analysis. The task involves performing binary text classification (positive or negative sentiment) using different recurrent neural network architectures. The models are trained and evaluated using the Urdu Sentiment Corpus dataset.

## Dataset

The dataset used for this project can be found [here](https://github.com/MuhammadYaseenKhan/Urdu-Sentiment-Corpus/blob/master/urdu-sentiment-corpus-v1.tsv). It contains Urdu tweets labeled with sentiment classes:
- P (Positive)
- N (Negative)

## Models Implemented

The following sequence-based deep learning models are implemented for sentiment analysis:
- RNN (Recurrent Neural Network)
- GRU (Gated Recurrent Unit)
- LSTM (Long Short-Term Memory)
- BiLSTM (Bidirectional Long Short-Term Memory)

## Hyperparameters

Each model is trained and evaluated with the following hyperparameters:
- Number of layers: 2 or 3
- Dropout rate: 0.3 or 0.7

This results in 4 different sets of parameters for each model.

## Evaluation Metrics

The models are evaluated based on the following metrics:
- Accuracy
- Precision
- Recall
- F-score

## Conclusion

Overall, the advanced models (GRU, LSTM, and BiLSTM) are expected to outperform the basic RNN model due to their ability to handle long-term dependencies more effectively. Among them, BiLSTM typically achieves the best results by leveraging information from both directions in a sequence. The hyperparameters need to be tuned carefully to balance model complexity and regularization to achieve optimal performance.


**Download the dataset:**
    Download the dataset from [here](https://github.com/MuhammadYaseenKhan/Urdu-Sentiment-Corpus/blob/master/urdu-sentiment-corpus-v1.tsv) and place it in the `data/` directory.

**Run the notebook:**
    Open `UrduSentimentAnalysis.ipynb` in Jupyter Notebook and run all the cells to train and evaluate the models.
   

## Acknowledgements

- The dataset used in this project is provided by [Muhammad Yaseen Khan](https://github.com/MuhammadYaseenKhan/Urdu-Sentiment-Corpus).
