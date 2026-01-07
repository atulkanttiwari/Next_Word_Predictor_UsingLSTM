# Next Word Predictor Using LSTM RNN

## Project Overview

This project develops a deep learning model for predicting the next word in a given sequence of words using Long Short-Term Memory (LSTM) networks. The model is trained on a corpus of Shakespeare's texts and deployed as a Streamlit web application for real-time predictions.

## Features

- **Data Collection**: Utilizes texts from Shakespeare's works (Hamlet, Macbeth, Julius Caesar) and other classic literature as the training dataset.
- **Data Preprocessing**: Tokenizes text, creates input sequences, and pads them for uniform input lengths.
- **Model Architecture**: Employs an LSTM-based neural network with embedding layers, dropout for regularization, and a softmax output layer.
- **Training**: Includes early stopping to prevent overfitting and validation on a test set.
- **Deployment**: Interactive Streamlit app allowing users to input text and receive next word predictions.
- **Evaluation**: Tests the model with example sentences to assess prediction accuracy.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd next-word-predictor-using-lstm-rnn
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the Jupyter notebook `experiemnts.ipynb` to preprocess data, build, train, and save the LSTM model and tokenizer.

### Running the Streamlit App

Execute the following command to start the web application:
```
streamlit run app.py
```

Open your browser and navigate to the provided local URL. Enter a sequence of words in the input field and click "Predict Next Word" to see the model's prediction.

## Dependencies

- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- TensorBoard
- Matplotlib
- Streamlit
- SciKeras
- IPyKernel
- NLTK

## Model Details

- **Input**: Sequence of words (padded to max sequence length).
- **Output**: Predicted next word with highest probability.
- **Architecture**:
  - Embedding layer (100 dimensions)
  - LSTM layer (150 units, return sequences)
  - Dropout (0.2)
  - LSTM layer (100 units)
  - Dense layer (softmax activation)

## Files

- `app.py`: Streamlit application for next word prediction.
- `experiemnts.ipynb`: Jupyter notebook containing data preprocessing, model training, and evaluation.
- `requirements.txt`: Python dependencies.
- `hamlet.txt`: Combined text corpus from Shakespeare's works.
- `next_word_lstm.keras`: Trained LSTM model.
- `tokenizer.pickle`: Pickled tokenizer for text processing.

## Contributing

Feel free to fork the repository and submit pull requests for improvements or additional features.
