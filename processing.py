import pandas as pd
import numpy as np
from scipy.fft import fft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Define global variables
window_size = 10

def data_processing(file_path):
    # Read data from CSV file
    df = pd.read_csv(file_path)
    
    # Extract features and targets
    X_raw = df.iloc[:, 1:].values  # Assuming the first column contains labels
    Y_raw = df.iloc[:, 0].apply(lambda x: eval(x)).values.tolist()  # Convert string coordinates to lists
    
    # Process data into input and target arrays
    X_processed = []
    Y_processed = []
    
    for i in range(len(X_raw) - window_size + 1):
        X_window = X_raw[i:i + window_size]
        
        # Apply FFT to each channel in the window
        X_window_processed = [np.abs(fft(channel)) for channel in X_window]
        
        X_processed.append(X_window_processed)
        Y_processed.append(Y_raw[i + window_size - 1])  # Target corresponds to the last label in the window
    Y_processed = np.array(Y_processed).reshape(-1, 2)  # Reshape to (num_samples, 2)

    return np.array(X_processed), Y_processed

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=32, input_shape=input_shape, return_sequences=True))
    model.add(Dense(2))  # Output layer with 2 units for (x, y) coordinates
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, Y_train):
    model = create_model(X_train.shape[1:])
    model.fit(X_train, Y_train, epochs=10, batch_size=32)  # Adjust epochs and batch size as needed
    return model

def main():
    # Load and preprocess data
    X, Y = data_processing("pixel2.csv")
    
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Train the model
    trained_model = train_model(X_train, Y_train)
    
    # Evaluate the model on the test set
    loss = trained_model.evaluate(X_test, Y_test)
    print("Test loss:", loss)

if __name__ == "__main__":
    main()
