
import pandas as pd


#print(tf.__version__)

train = pd.read_csv('train.csv')
test =  pd.read_csv('test.csv')

def outcome_to_numeric(x):
    if x=='Inactive':
        return 0
    if x=='Active':
        return 1

train['label'] = train['Outcome'].apply(outcome_to_numeric)
test['label'] = test['Outcome'].apply(outcome_to_numeric)
test.head()

train=train.drop('Outcome', axis=1)
test=test.drop('Outcome', axis=1)

x_train = train.drop('label', axis=1)
y_train = train['label']

x_test = test.drop('label', axis=1)
y_test = test['label']

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


model = ExtraTreesClassifier()
model.fit(x_train, y_train)

feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

x_train = x_train[['WBN_GC_L_0.75', 'WBN_GC_H_0.50', 'WBN_EN_H_0.75', 'XLogP', 'WBN_EN_H_0.50']]

x_test = x_test[['WBN_GC_L_0.75', 'WBN_GC_H_0.50', 'WBN_EN_H_0.75', 'XLogP', 'WBN_EN_H_0.50']]

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.noise import AlphaDropout
from keras import optimizers
from keras import layers
from tensorflow.keras import *


model = Sequential()
model.add(Dense(45, input_dim=5, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(45, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(45, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

opt = optimizers.Adadelta(lr=.01)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])