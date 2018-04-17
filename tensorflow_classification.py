import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#step 1 load the data
dataframe  = pd.read_csv('data.csv')
dataframe = dataframe.drop(['index' , 'price', 'sq_price' ], axis = 1) #drop the index column
dataframe = dataframe[0:10] # to read dataframe of rows from 0 to 10

#add a new column of labels for classification
# 1 is good buy and 0 is bad buy
dataframe.loc[: , ('y1')] = [1 ,1  , 1 , 0 , 0,1 , 0  , 1 , 1 , 0]

dataframe.loc[: , ('y2')] = dataframe['y1'] == 0 # y2 is a negation of y1
dataframe.loc[: , ('y2')] = dataframe['y2'].astype(int) #convert true false to 1 and 0

#step 3 prepare data for tensorflow
inputX = dataframe.loc[: , ['area, bathroom']].as_matrix() #converting features to input tensor

inputY = dataframe.loc[: , ['y1 , y2']].as_matrix() #convert labels to input tensors

#step 4 write our hyperparameters

learning_rate = 0.0001
training_epochs  = 2000
display_step = 50
n_samples = inputY.size

#step 5 create our computational graph / neural network
#placeholders are gateways for data into our computational graph
x = tf.placeholder(tf.float32 , [None , 2]) #None means we can specify any number of examples into the tensorflow placeholder , 2 because we are inputting 2 features

#create weights
# 2x 2 float matrix
#Variables hold and update parameters
w = tf.Variable(tf.zeros([2,2]))

# adding biases
b = tf.Variable(tf.zeros([2])) # 2 because we have two inputs
#biases help our model fit better
# for y  = mx + b that is linear regression

#multiply our weights by  our inputs
y_values = tf.add(tf.matmul(x , W), b)

#apply softmax to values we just created
#softmax is our activation function
y = tf.nn.softmax(y_values)

#feed in a matrix of labels
y_ = tf.placeholder(tf.float32 , [None , 2])



# perform training
#create our cost function , mean squared error
#reduce_sum computes the sum of elements across dimensions of a tensor

cost = tf.reduce_sum(tf.pow(y_ - y , 2))/(2 * n_samples)

#gradient descent
#optimizer function
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#initialize variables and tensorflow session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#training loop
for i in range(training_epochs):
    sess.run(optimizer , feed_dict = {x : inputX , y_ : inputY})

    #loggs for training
    if (i) % display_step == 0:
        cc = sess.run(cost , feed_dict = {x : inputX , y_ : inputY})
        print ("Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc))

print ('Optimization finished')
training_cost = sess.run(cost , feed_dict = {x : inputX , y_ : inputY})
print ('Training Cost = ' , training_cost , 'W= ' , sess.run(W) , 'b=' , sess.run(b))

sess.run(y, feed_dict = {x:inputX})

             





