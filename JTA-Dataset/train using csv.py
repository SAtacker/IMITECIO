import pandas as pd
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

fact = 10000
path = os.getcwd()
path += r"\csv"
num_files = int(len(os.listdir(path))/fact)
# num_files = 100000
x_train = np.empty([num_files,12,2])
y_train = np.empty([num_files,12,3])
cnt = 0
# pbad = tqdm(total=num_files*12)
print("Loading train set")
with tqdm(total=num_files) as bar:
    for fname in os.listdir(path):
        df = pd.read_csv(os.path.join(path,fname))
        # print(df.iloc[0,:])
        for i in range(0,12):
            # x_train.append(df.iloc[i,1:3].values)
            x_train[cnt][i] = df.iloc[i,1:3].to_numpy()
            y_train[cnt][i] = df.iloc[i,3:].to_numpy()
        # break
        cnt+=1
        bar.update()
        if cnt == num_files :
            break

########## Remember that the x2d and y2d are of resolution 1920 x 1080 ##################

# pbad.close()
# print(y_train)
x_train /= np.linalg.norm(x_train)
# y_train /= np.linalg.norm(y_train)
# x_train = np.expand_dims(x_train, axis=0)
# y_train = np.expand_dims(y_train, axis=0)

classifier = SVC(kernel = 'rbf', C = 0.1, gamma = 0.1)
x_train = x_train.reshape(num_files,24)
y_train = y_train.reshape(num_files,36)
classifier.fit(x_train, y_train)
# model = keras.Sequential([
#         keras.Input(shape=(12,2)),
#         keras.layers.Dense(2048, activation=keras.layers.PReLU(alpha_initializer='ones', alpha_regularizer=None, alpha_constraint=None, shared_axes=None), name="layer1"),
#         keras.layers.Dense(1024, activation=keras.layers.PReLU(alpha_initializer='ones', alpha_regularizer=None, alpha_constraint=None, shared_axes=None), name="layer2"),
#         keras.layers.Dense(512, activation=keras.layers.PReLU(alpha_initializer='ones', alpha_regularizer=None, alpha_constraint=None, shared_axes=None), name="layer3"),
#         keras.layers.Dense(256, activation=keras.layers.PReLU(alpha_initializer='ones', alpha_regularizer=None, alpha_constraint=None, shared_axes=None), name="layer4"),
#         keras.layers.Dense(3,activation=keras.layers.PReLU(alpha_initializer='ones', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)),
#     ])

# model = keras.Sequential([
#         keras.Input(shape=(12,2)),
#         keras.layers.Dense(2048, activation="relu", name="layer1"),
#         keras.layers.Dense(1024, activation="softmax", name="layer2"),
#         keras.layers.Dense(512, activation="relu", name="layer3"),
#         keras.layers.Dense(256, activation="tanh", name="layer4"),
#         keras.layers.Dense(3),
#     ])

# model.compile(
#     optimizer="adadelta",
#     loss="squared_hinge",
#     metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
# )

# hist = model.fit(x_train, y_train, epochs=100)
# # model.evaluate(x_test,y_test)
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['loss'])
# plt.show()
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

num_files = int(len(os.listdir(path))/fact)
# num_files = 100000
x_test = np.empty([num_files,12,2])
y_test = np.empty([num_files,12,3])
cnt = 0
inn = 0
print("Loading test set")
with tqdm(total=num_files) as bar:
    for fname in os.listdir(path):
        inn += 1
        if inn < 10000 :
            cnt = 0
            continue
        df = pd.read_csv(os.path.join(path,fname))
        # print(df.iloc[0,:])
        for i in range(0,12):
            # x_train.append(df.iloc[i,1:3].values)
            x_test[cnt][i] = df.iloc[i,1:3].to_numpy()
            y_test[cnt][i] = df.iloc[i,3:].to_numpy()
        # break
        cnt+=1
        bar.update()
        if cnt == num_files :
            break

########## Remember that the x2d and y2d are of resolution 1920 x 1080 ##################

# pbad.close()
# print(y_train)
x_test /= np.linalg.norm(x_test)
# y_test /= np.linalg.norm(y_test)
# x_test = np.expand_dims(x_test, axis=0)
# y_test = np.expand_dims(y_test, axis=0)

# model.evaluate(x_test,y_test)
x_test = x_test.reshape(num_files,24)
y_pred = classifier.predict(x_test)

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#  Model Precision: what percentage of positive tuples are labeled as such?

print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))