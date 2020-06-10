import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

fact = 1000
path = os.getcwd()
path += r"\csv"
num_files = int(len(os.listdir(path))/fact)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
model.compile(
    optimizer="adadelta",
    loss="squared_hinge",
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)
print("Loaded model from disk")

# num_files = int(len(os.listdir(path))/fact)
# num_files = 100000
x_test = np.empty([num_files,12,2])
y_test = np.empty([num_files,12,3])
cnt = 0
inn = 0
print("Loading test set")
with tqdm(total=num_files) as bar:
    for fname in os.listdir(path):
        inn += 1
        if inn < 70000 :
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


model.evaluate(x_test,y_test)
print(model.predict(x_test[:1]))
print(y_test[:1])