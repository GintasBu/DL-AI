import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [ series[i:i+window_size] for i in range(len(series)-window_size) ]
    y = [series[i+window_size] for i in range(len(series)-window_size) ]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model=Sequential()
    model.add(LSTM(units=5, input_shape=[window_size, 1] ))
    model.add(Dense(1, activation=None))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    import re
    punctuation = ['!', ',', '.', ':', ';', '?']
    charak=set(text)
    sym=re.findall(r'[^a-z]', ''.join(charak))
    punctuation.append(' ')
    charak_to_remove=[]
    for s in sym:
        if s not in punctuation:
            charak_to_remove.append(s)
    for s in charak_to_remove:
        text = text.replace(s,' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    
    inputs = [ text[i*step_size:i*step_size+window_size] for i in range(int(np.ceil((len(text)-window_size)/step_size)))]
    outputs = [text[i*step_size+window_size] for i in range(int(np.ceil((len(text)-window_size)/step_size)) )]
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model=Sequential()
    model.add(LSTM(units=200, input_shape=[window_size, num_chars]))
    model.add(Dense(num_chars, activation='softmax'))
    return model
    
