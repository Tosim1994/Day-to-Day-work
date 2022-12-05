import numpy as np
import numpy
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.model_selection import *
import pandas as pd

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


def numerical_class(i):
    if i == 'Inactive':
        return 0
    else:
        return 1


df_train['label'] = df_train['Outcome'].apply(numerical_class)
df_test['label'] = df_test['Outcome'].apply(numerical_class)


X_train = df_train.drop(['Outcome','label'],axis='columns').values
y_train = df_train[['label']].values


X_test = df_test.drop(['Outcome','label'],axis='columns').values

y_test = df_test[['label']].values


#print(y_test)
#print(X_test)

for i in np.arange(1,2,1):
    #model_ml_emir = ExtraTreeClassifier(criterion="gini", splitter="random", max_features="auto", random_state=9)

    from sklearn.neighbors import *

    #58 random #4 estimator

    model_ml_emir = BaggingClassifier(base_estimator=ExtraTreesClassifier(n_estimators=4,random_state=37),
                                      n_estimators=9,
                                      random_state=37,
                                      bootstrap=True,warm_start=True,verbose=0)

    model_ml_emir.fit(X_train, y_train)

    prediction = model_ml_emir.predict(X_test)

    accuracy_score(y_pred=prediction, y_true=y_test)

    print("X",i)

    print("Machine Learning Software is the Accuracy Score: {0} "
          .format(accuracy_score(y_pred=prediction, y_true=y_test)))
    print("Machine Learning Software is the Precision Score: {0} "
          .format(precision_score(y_pred=prediction, y_true=y_test)))
    print("Machine Learning Software is the Recall Score: {0} "
          .format(recall_score(y_pred=prediction, y_true=y_test)))
    print("Machine Learning Software is the F1 Score: {0} "
          .format(f1_score(y_pred=prediction, y_true=y_test)))