#Import all necessary libraries
import random
import numpy as np
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Concatenate, Dot, Lambda, Input, Conv2D, MaxPooling2D, Dropout
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
import matplotlib.pyplot as plt

#Load-Data

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))

tr_ints, t_ints = [0,1,2,4,5,9], [3,6,7,8]

# Creates a boolean mask for each set
tr_set = [x in tr_ints for x in Y]
t_set = [x in t_ints for x in Y]

x_tr, x_t = X[tr_set], X[t_set]
y_tr, y_t = Y[tr_set], Y[t_set]

# Split 80% to training and 20% to test
split = int(len(x_tr) * 0.8)
x_t2, x_tr = x_tr[split::], x_tr[:split:]
y_t2, y_tr = y_tr[split::], y_tr[:split:]

# Function to output sets of matching Pairs and non-matching Pairs to train our model
def make_train_pairs(x, y):
    train_labels = np.unique(y)
    num_classes = len(train_labels)
    digit_indices = [np.where(y==i)[0] for i in train_labels]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        #matching label
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = np.random.choice(digit_indices[np.where(train_labels == label1)])
        x2 = x[idx2]

        pairs += [[x1,x2]]
        labels += [0]

        #not matching label

        label2 = train_labels[random.randint(0, num_classes - 1)]
        while label2 == label1:
            label2 = train_labels[random.randint(0, num_classes - 1)]

        idx2 = random.choice(digit_indices[train_labels.index(label2)])
        x2 = x[idx2]

        pairs += [[x1,x2]]
        labels += [1]

    return np.array(pairs), np.array(labels)

pairs_train, pairslabel = make_train_pairs(x_tr, y_tr)
print(pairs_train.shape, pairslabel.shape)

# def make_test_pairs(x, y):
#     num_classes = 4
#     test_labels = [3,6,7,8]
#     digit_indices = [np.where(y==i)[0] for i in (3,6,7,8)]
#
#     pairs = []
#     labels = []
#
#     for idx1 in range(len(x)):
#         #matching label
#         x1 = x[idx1]
#         label1 = y[idx1]
#         idx2 = random.choice(digit_indices[test_labels.index(label1)])
#         x2 = x[idx2]
#
#         pairs += [[x1, x2]]
#         labels += [0]
#
#         #not matching label
#
#         label2 = test_labels[random.randint(0, num_classes - 1)]
#         while label2 == label1:
#             label2 = test_labels[random.randint(0, num_classes - 1)]
#
#         idx2 = random.choice(digit_indices[test_labels.index(label2)])
#         x2 = x[idx2]
#
#         pairs += [[x1,x2]]
#         labels += [1]
#
#     return np.array(pairs), np.array(labels)


def euclidean_distance(vects):
    x,y = vects
    sum_square = K.sum(K.square(x-y), axis= 1, keepdims= True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def base_model(input_shape):
    input_layer = Input(shape=input_shape)
    model = Conv2D(32, kernel_size=(3, 3), activation="relu")(input_layer)
    model = Conv2D(64, (3, 3), activation="relu")(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.25)(model)
    model = Flatten()(model)
    model = Dense(128, activation="relu")(model)
    model = Dropout(0.5)(model)
    model = Dense(10, activation="relu")(model)

    return Model(input_layer, model)


def compute_accuracy(y_ground_truth, y_pred):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_ground_truth)


def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def accuracy_cust(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# pairs_train, labels_train = make_train_pairs(x_train, y_train)
# pairs_test, labels_test = make_test_pairs(x_test, y_test)
#
# input = Input((28,28))
# x = Flatten()(input)
# x = Dense(128, activation='relu')(x)
# dense = Model(input, x)
# input1 = Input((28,28))
# input2 = Input((28,28))
#
# dense1 = dense(input1)
# dense2 = dense(input2)
#
# merge_layer = Lambda(euclidean_distance)([dense1, dense2])
# # mod1 = Sequential()
# # mod1.add(Flatten(input_shape=(28,28)))
# # mod1.add(Dense(128, activation='relu'))
# #
# # mod2 = Sequential()
# # mod2.add(Flatten(input_shape=(28,28)))
# # mod2.add(Dense(128, activation='relu'))
# #
# dense_layer = Dense(1, activation='sigmoid')(merge_layer)
#
# model = Model(inputs=[input1, input2], outputs=dense_layer)
# model.summary()
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
#
# model.fit([pairs_train[:,0], pairs_train[:,1]], labels_train[:], epochs=5)
