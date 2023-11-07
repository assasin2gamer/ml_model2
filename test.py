import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.fft import fft
import matplotlib.pyplot as plt
import random
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import re


window_size = 10
test_Y = []
test_X = []

def data_generator(file_path, chunk_size):
    csv_reader = pd.read_csv("./pixel2.csv", chunksize=100)
    Y_chunk = []
    X_chunk = []
    for chunk in csv_reader:
        for index, row in chunk.iterrows():
            coordinates_match = re.match(r"\((.*),(.*)\)", row.iloc[0])
            if coordinates_match:
                x = float(coordinates_match.group(1))
                y = float(coordinates_match.group(2))
                coordinates_array = [x, y]
                Y_chunk.append(coordinates_array)
            X_chunk.append(row.iloc[1:].values.tolist())
        processing(X_chunk, Y_chunk)
        X_chunk = []
        Y_chunk = []

        
def processing(X_chunk, Y_chunk):
    global window_size

    processed_X = []

    for i in range(len(X_chunk) - 5 + 1):
        if i > 5:
            X_window = X_chunk[i - window_size:i + window_size]
            for channels in X_window:
                channels = np.abs(fft(channels))
            flat_data = (np.array(X_window).flatten().tolist())
            if (len(flat_data) == 280):
                processed_X.append(flat_data)
  
        else:
            X_window = X_chunk[i:i + window_size + window_size]
            for channels in X_window:
                channels = np.abs(fft(channels))
            flat_data = (np.array(X_window).flatten().tolist())
            if (len(flat_data) == 280):
                processed_X.append(flat_data)

    with open("example.txt", "a") as file:
        file.write(str(X_chunk[0]) + " "+ str(Y_chunk[0]) )
    if len(processed_X) == 87:
        train(processed_X, Y_chunk[:len(processed_X)])
        #print("hello")
    



def defineModel():
    global model
    model = keras.Sequential()
    model.add(SimpleRNN(units=32, activation='relu', input_shape=(280, 1)))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mean_squared_error')
    

def train(X_chunk, Y_chunk):
    global model
    model.fit(X_chunk, Y_chunk, epochs=10, batch_size=1, verbose=1)
    verify(test_X, test_Y)
    model.save("my_model.h5")





def verify(test_X, test_Y):
    global model
    prediction = model.predict(test_X)

    for count in range(len(prediction)): 
        print("prediction :" + str(prediction[count]) + " Actual: " + str(test_Y[count]))

    
def get_test():
    global test_X
    global test_Y
    csv_reader = pd.read_csv("./test.csv", chunksize=100)
    Y_chunk = []
    X_chunk = []
    for chunk in csv_reader:
        for index, row in chunk.iterrows():
            coordinates_match = re.match(r"\((.*),(.*)\)", row.iloc[0])
            if coordinates_match:
                x = float(coordinates_match.group(1))
                y = float(coordinates_match.group(2))
                coordinates_array = [x, y]
                Y_chunk.append(coordinates_array)
            X_chunk.append(row.iloc[1:].values.tolist())
    
    processed_X = []


    for i in range(len(X_chunk) - 5 + 1):
        if i > 5:
            X_window = X_chunk[i - window_size:i + window_size]
            for channels in X_window:
                channels = np.abs(fft(channels))
            flat_data = (np.array(X_window).flatten().tolist())
            if (len(flat_data) == 280):
                processed_X.append(flat_data)
  
        else:
            X_window = X_chunk[i:i + window_size + window_size]
            for channels in X_window:
                channels = np.abs(fft(channels))
            flat_data = (np.array(X_window).flatten().tolist())
            if (len(flat_data) == 280):
                processed_X.append(flat_data)

    
    test_Y = Y_chunk 
    test_X = processed_X

    #print(test_X)
    #print(test_Y)




def main():
    global model
    get_test()
    defineModel()
    data_generator("./pixel.csv", 100)

main()


        
