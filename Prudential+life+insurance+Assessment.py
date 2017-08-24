
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# In[ ]:


train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')


# In[ ]:


train.head()


# In[ ]:


train.dtypes


# In[ ]:


train.corr()


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

correlation = train.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

plt.title('Correlation between different fearures')


# In[ ]:


coeff_df = pd.DataFrame(datas.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Correlation"] = pd.Series(logis.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


train['p2char']=train.Product_Info_2.str[0]
train['p2num']=train.Product_Info_2.str[1]


# In[ ]:


train.drop('Product_Info_2', axis=1,inplace=True)


# In[ ]:


train['p2char']=pd.factorize(train.p2char)[0]+1


# In[ ]:


train[['p2num']] = train[['p2num']].astype(float)


# In[ ]:


y=train['Response']


# In[ ]:


drop_columns=['Id','Response']


# In[ ]:


train.drop(drop_columns,axis=1,inplace=True)


# In[ ]:


train.fillna(train.mean(), inplace=True)


# In[ ]:


X=train


# In[ ]:


#log reg to check the coefficients

import timeit
start_time = timeit.default_timer()

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)

# check the accuracy on the training set
model.score(X, y)
elapsed = timeit.default_timer() - start_time


# In[ ]:


coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(model.coef_))], axis = 1)


# In[ ]:


coefficients


# In[ ]:


## imputing function 
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
    ##
    
    #imputing function
    xt = DataFrameImputer().fit_transform(X)


# In[ ]:


# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=0)


# In[ ]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression()
logis.fit(X_train, y_train)
logis_score_train = logis.score(X_train, y_train)
print("Training score: ",logis_score_train)
logis_score_test = logis.score(X_test, y_test)
print("Testing score: ",logis_score_test)


# In[ ]:


#random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_score_train = rfc.score(x_train, y_train)
print("Training score: ",rfc_score_train)
rfc_score_test = rfc.score(x_test, y_test)
print("Testing score: ",rfc_score_test)


# In[ ]:


# predict class labels for the test set
predicted = model2.predict(X_test)
print (predicted)

# generate class probabilities
probs = model2.predict_proba(X_test)
print (probs)


# generate evaluation metrics
print(metrics.accuracy_score(y_test, predicted)) 
#print(metrics.roc_auc_score(y_test, predicted))


# In[ ]:


#SVM
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
svm_score_train = svm.score(X_train, y_train)
print("Training score: ",svm_score_train)
svm_score_test = svm.score(X_test, y_test)
print("Testing score: ",svm_score_test)



# In[ ]:


#kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_score_train = knn.score(X_train, y_train)
print("Training score: ",knn_score_train)
knn_score_test = knn.score(X_test, y_test)
print("Testing score: ",knn_score_test)


# In[ ]:


#decision tree

from sklearn import tree

dt = tree.DecisionTreeClassifier()

dt.fit(X_train, y_train)

dt_score_train = dt.score(X_train, y_train)
 
print("Training score: ",dt_score_train)

dt_score_test = dt.score(X_test, y_test)

print("Testing score: ",dt_score_test)


# In[ ]:


#Model comparison
models = pd.DataFrame({
        'Model'          : ['Logistic Regression', 'SVM', 'kNN', 'Decision Tree', 'Random Forest'],
        'Training_Score' : [logis_score_train, svm_score_train, knn_score_train, dt_score_train, rfc_score_train],
        'Testing_Score'  : [logis_score_test, svm_score_test, knn_score_test, dt_score_test, rfc_score_test]
    })
models.sort_values(by='Testing_Score', ascending=False)


# In[ ]:


#hyper parametertuning for random forest

from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)


rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)
print (CV_rfc.best_params_)


# In[ ]:


#random forest with tuned hyper parameters

from sklearn.ensemble import RandomForestClassifier

rfc= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='sqrt', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=700, n_jobs=2,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

rfc.fit(X_train, y_train)
rfc_score_train = rfc.score(X_train, y_train)
print("Training score: ",rfc_score_train)
rfc_score_test = rfc.score(X_test, y_test)
print("Testing score: ",rfc_score_test)


# In[ ]:


#Stacking Logistic, decision tree and SVM

# SklearnVoting Ensemble for Classification

import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

A = X
B = y
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, A, B, cv=kfold)
print(results.mean())


# In[ ]:


# First XGBoost model 

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

xgb_score_train = model.score(X_train, y_train)

print("Training score: ",xgb_score_train)

xgb_score_test = model.score(X_test, y_test)

print("Testing score: ",xgb_score_test)

# plot feature importance
plot_importance(model)
pyplot.show()

# make predictions for test data

#y_pred = model.predict(X_test)

#predictions = [round(value) for value in y_pred]
# evaluate predictions
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


#Multi layer perceptron

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)


# In[ ]:




# baseline model
def baseline_model():
    model = Sequential()
    #model.add(Dense(10, input_dim=127, activation='relu'))
    #model.add(Dense(8, activation='softmax'))
    model.add(Dense(input_dim=127, output_dim=200, init="uniform", activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=200, output_dim=200, init="uniform", activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=200, output_dim=9, init="uniform", activation="softmax"))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[ ]:


estimator = KerasClassifier(build_fn=baseline_model, epochs=2, batch_size=5, verbose=0)


# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.optimizers import Adam

model = Sequential()

model.add(Dense(input_dim=127, output_dim=200, init="uniform", activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(input_dim=200, output_dim=200, init="uniform", activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(input_dim=200, output_dim=9, init="uniform", activation="softmax"))


# In[ ]:


adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# In[ ]:


response_transformed = np_utils.to_categorical(y)
print(response_transformed.shape)



# In[ ]:


model.fit(X.values, response_transformed, nb_epoch=5, batch_size=50)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, response_transformed)
model.fit(X_train, y_train, nb_epoch=50, batch_size=50)


# In[ ]:


#use extraTreesClassifier for extracting feature importances this was referenced from kaggle


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
rf = ExtraTreesClassifier(n_estimators=300,
                              random_state=0)
rf.fit(X, y)


# In[ ]:


importances =pd.DataFrame({'features' :X.columns,
                           'importances' : rf.feature_importances_})
importances.sort_values(by = 'importances', ascending = False).head(20)


importances.sort_values(by = 'importances', ascending = False).tail(20)


# In[ ]:


plot importances

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

importances.sort_values(by = 'importances', ascending = True, inplace = True)
val = importances.importances*100  
pos = np.arange(importances.shape[0])+.5 

plt.figure(figsize = (13,28))
plt.barh(pos,val, align='center')
plt.yticks(pos, importances.features.values)
plt.xlabel('Importances')
plt.title('Features importances')
plt.grid(True)


# In[ ]:


#cumsum of importances

importances.sort_values(by = 'importances', ascending = False, inplace = True)

importances['cumul'] = np.cumsum(importances.importances, axis = 0)


# In[ ]:


importances.sort_values(by = 'importances', ascending = True, inplace = True)

val = importances.cumul*100    # the bar lengths
pos = np.arange(importances.shape[0])+.5 

plt.figure(figsize = (13,28))
plt.barh(pos,val, align='center')
plt.yticks(pos, importances.features.values)
plt.xlabel('Importances')
plt.title('Features importances')
plt.grid(True)


# In[ ]:


#Variables ro remove to get X % of importances 

X = 95

importances.features[importances.cumul>X/100].values


# In[ ]:


#below are other implementation of algorithms 


# In[ ]:


from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy
import numpy as np 
import pandas as pd 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# In[ ]:


seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:


# CV model
model = XGBClassifier()
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X1, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:


import matplotlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier 


clf = RandomForestClassifier(n_estimators=250, n_jobs=-1)
# we use a BaggingClassifier to make 5 predictions, and average
# beacause that's what CalibratedClassifierCV do behind the scene
# and we want to compare things fairly
clfbag = BaggingClassifier(clf, n_estimators=5)
clfbag.fit(X_train, y_train)
# make predictions for test data
y_pred = clfbag.predict(X_test)

predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#ypreds = clfbag.predict_proba(X_test)
#print "%.2f"% % log_loss(y_test, ypreds, eps=1e-15, normalize=True)


# In[ ]:


get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from brew.base import Ensemble, EnsembleClassifier
from brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
from brew.combination.combiner import Combiner



# In[ ]:


# Initializing Classifiers
clf1 = LogisticRegression(random_state=0)
clf2 = RandomForestClassifier(random_state=0)
clf3 = SVC(random_state=0, probability=True)

# Creating Ensemble
ensemble = Ensemble([clf1, clf2, clf3])
eclf = EnsembleClassifier(ensemble=ensemble, combiner='mean')

# Creating Stacking
layer_1 = Ensemble([clf1, clf2, clf3])
layer_2 = Ensemble([sklearn.clone(clf1)])

stack = EnsembleStack(cv=3)

stack.add_layer(layer_1)
stack.add_layer(layer_2)

sclf = EnsembleStackClassifier(stack, combiner=Combiner('mean'))

clf_list = [clf1, clf2, clf3, eclf, sclf]
lbl_list = ['Logistic Regression', 'Random Forest', 'RBF kernel SVM', 'Ensemble', 'Stacking']


# In[ ]:



from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from brew.base import Ensemble, EnsembleClassifier
from brew.combination.combiner import Combiner 
    
clf1 = LogisticRegression(random_state=0)
clf2 = RandomForestClassifier(random_state=0)
clf3 = SVC(random_state=0, probability=True)
clf4=  GradientBoostingClassifier( learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)
        
    # create your Ensemble clf1 can be an EnsembleClassifier object too
ens = Ensemble(classifiers=[clf1, clf2, clf3,clf4]) 
     
    # create your Combiner (combination rule)
    # it can be 'min', 'max', 'majority_vote' ...
cmb = Combiner(rule='mean')
     
    # and now, create your Ensemble Classifier
ensemble_clf = EnsembleClassifier(ensemble=ens, combiner=cmb)
     
    # assuming you have a X, y data you can use
ensemble_clf.fit(X1, y)
ensemble_clf.predict(X1)
ensemble_clf.predict_proba(X1)
     


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




