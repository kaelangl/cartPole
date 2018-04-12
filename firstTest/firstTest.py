import gym
import keras
from keras.models import Sequential
from keras.layers import Dense

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()

def initial_population():

    for episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

def neural_network():

    ann = Sequential()

    # add hidden layer
    # by adding the first hidden layer it automatically gives us how many inputs we have
    # based on the input
    ann.add(Dense(activation="relu", input_dim=11, units=12, kernel_initializer="glorot_uniform"))

    # Add second hidden layer, don't need to say the inputs anymore
    ann.add(Dense(activation="relu", units=24, kernel_initializer="glorot_uniform"))

    # Add third hidden layer
    ann.add(Dense(activation="relu", units=12, kernel_initializer="glorot_uniform"))

    # Adding output layer
    # if you had more than one dependent variable, like if you onehotencoded it
    # you would have units as t - 1, where t is number of variables
    # and you would ave the activation function as softmax
    ann.add(Dense(activation="sigmoid", units=12, kernel_initializer="glorot_uniform"))

    # Compiling the ANN, which means applying stochastic gradient descent to it
    # using logarithmic loss as loss function, if you have a binary output then
    # algorim is called binary_crossentropy; categorical is categorical_crossentropy
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

initial_population()