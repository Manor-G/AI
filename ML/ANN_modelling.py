import os
import pandas as pd
import numpy as np

# Load the dataset
filepath = r"/Users/_/Desktop/My AI/dataset"
file = os.path.join(filepath, "Churn_Modelling.csv")
dataset = pd.read_csv(file)

print(dataset.head(10))

# Set x and y
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode the geography and gender section
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder= LabelEncoder()
X[:, 2] = labelencoder.fit_transform(X[:, 2])
print(X)

ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.int)
print(pd.DataFrame(X).head(10))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_valid = sc.transform(X_valid)

print(X_train,"\n")
print(X_test)

# Build the sequantial neural network

"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=15, activation="relu",input_dim=12))
model.add(Dense(units=30, activation="relu"))
model.add(Dense(units=25, activation="relu"))
model.add(Dense(units=1,activation="sigmoid"))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=10, epochs=100)
print(model.summary())
print(history.epoch)
print(history.history)

y_pred = model.predict(X_test)
print(y_pred)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm)


"""

# Build the functional artificial neural network

"""
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from keras import Model
input_ = Input(shape =(12,))
hidden1 = Dense(units=15, activation='relu')(input_)
hidden2 = Dense(units=30, activation='relu')(hidden1)
hidden3 = Dense(units=30, activation='relu')(hidden2)
hidden4 = Dense(units=25, activation='relu')(hidden3)
output = Dense(units=1, activation="sigmoid")(hidden4)
model = Model(inputs=[input_], outputs=[output])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Used earlystopping to stop the epoch running once the val_loss have no change could save the model at the same time
# Could save the optimal model at the same time by using callback as well
# In this case, the epoch only run 22-26 times, and it reaches the no val_loss situations. and the val_acc was 0.8575
# The cal_accuracy can be reached to 99.8 with 2000 and more epochs
# So the question would be: what value do we wanna to optimized

from tensorflow.keras import callbacks
# Monitor default val_loss, options with loss, accuracy, and val_accuracy.
early_stopping_cb = callbacks.EarlyStopping(patience=10, monitor='accuracy', restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=10, epochs=10000, callbacks=[early_stopping_cb])

print(model.summary())
print(history.epoch)
print(history.history)

y_pred = model.predict(X_test)
print(y_pred)

"""
import time
start_time = time.time()

# Tuning the model by using GridsearchCV

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
    classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=25, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [10, 25, 32],
              'epochs': [200, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters, best_accuracy)

print("---%s seconds___" % (time.time()-start_time))
