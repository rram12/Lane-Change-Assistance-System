from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def build_model(lr, n_actions, input_dim, fc1_dim, fc2_dim):
    model = Sequential()
    model.add(Dense(input_dim[0] * input_dim[1], input_dim= 1, activation="relu"))
    model.add(Dense(units=fc1_dim, activation='relu'))
    model.add(Dense(units=fc2_dim, activation='relu'))
    model.add(Dense(n_actions, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    
    return model

def save_plot(x, y1, y1label, fname, xlabel, ylabel, title, y2label=None, y3label=None, y2=None, y3=None):
    fig = plt.figure()
    x_axis = x
    plt.plot(x_axis, y1, label=y1label)
    if y2 is not None:
        plt.plot(x_axis, y2, label=y2label)
    if y3 is not None:
        plt.plot(x_axis, y3, label=y3label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    fig.savefig(fname, dpi=fig.dpi)