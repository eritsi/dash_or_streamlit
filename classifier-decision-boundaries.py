# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Feature space and classification boundaries
#
# While dtreeviz is dedicated primarily to showing decision trees, we have also provided a way to show the decision boundaries for one- and two- variable classifiers. The `decision_boundaries()` function will work with any model that answers method `predict_proba()` and with Keras, for which we provided a special adapter (since that method is deprecated).

# +
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_wine, load_digits, \
                             load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
# %config InlineBackend.figure_format = 'svg'  # Looks MUCH better than retina
# # %config InlineBackend.figure_format = 'retina'

from sklearn import tree

import dtreeviz
from dtreeviz import decision_boundaries
# -

# ## Two-variable classifications

# ### Wine data set

wine = load_wine()
X = wine.data
X = X[:,[12,6]]
y = wine.target

X.shape

# +
rf = RandomForestClassifier(n_estimators=50, min_samples_leaf=20, n_jobs=-1)
rf.fit(X, y)

fig,axes = plt.subplots(1,2, figsize=(8,3.8), dpi=300)
decision_boundaries(rf, X, y, ax=axes[0], fontname="monospace",
       feature_names=['proline', 'flavanoid'])
decision_boundaries(rf, X, y, ax=axes[1], fontname="monospace",
       # show classification regions not probabilities
       show=['instances', 'boundaries', 'misclassified'], 
       feature_names=['proline', 'flavanoid'])
plt.show()
# -

fig,ax = plt.subplots(1,1, figsize=(4,3))
decision_boundaries(rf, X, y, ax=ax, fontname="monospace",
       ntiles=20,
       tile_fraction=1.0, # make continuous, no white borders between tiles
       markers=['o','X','s'], # use different markers
       feature_names=['proline', 'flavanoid'],
       colors={'scatter_marker_alpha':.5})

# ### Titantic

# +
df = pd.read_csv("https://raw.githubusercontent.com/parrt/dtreeviz/master/data/titanic/titanic.csv")

df['Sex'] = np.where(df['Sex']=='male', 0, 1)
# -

X, y = df.drop(['Survived','Name','Ticket','Cabin','Embarked'], axis=1), df['Survived']
X['Age_na'] = X['Age'].isna()
X['Age'] = X['Age'].fillna(X['Age'].median(skipna=True))
X = X[['Age','Fare']]

rf = RandomForestClassifier(n_estimators=20, min_samples_leaf=3, n_jobs=-1)
rf.fit(X.values, y.values)

decision_boundaries(rf, X, y, ntiles=50, fontname="monospace", 
       binary_threshold=.5,
       markers=['X','s'],
       feature_names=['Age','Fare'])

# Hideous colors for Oliver Zeigermann
decision_boundaries(rf, X.values, y, fontname="monospace",
             markers=['X', 's'],
             feature_names=['Age', 'Fare'],
             colors={'class_boundary': 'red',
                     'classes':
                         [None,  # 0 classes
                          None,  # 1 class
                          ["#73ADD2", "#FEE08F"],  # 2 classes
                          ]})

# ### Cancer

# +
cancer = load_breast_cancer()

df = pd.DataFrame(data=cancer.data)
df.columns = [f'f{i}' for i in range(df.shape[1])]
df['y'] = cancer.target
# -

X, y = df.drop('y',axis=1), df['y']

# +
X = df[['f27','f22']]

rf = RandomForestClassifier(n_estimators=30, min_samples_leaf=5, n_jobs=-1)
rf.fit(X.values, y.values)
# -

decision_boundaries(rf, X.values, y, fontname="monospace",
             markers=['X','s'],
             feature_names=['f27', 'f22'],
             dot_w=20)


# ### Synthetic data set

def smiley(n = 1000):
    # mouth
    x1 = np.random.normal(1.0,.2,n).reshape(-1,1)
    x2 = np.random.normal(0.4,.05,n).reshape(-1,1)
    cl = np.full(shape=(n,1), fill_value=0, dtype=int)
    d = np.hstack([x1,x2,cl])
    data = d
    
    # left eye
    x1 = np.random.normal(.7,.2,n).reshape(-1,1)
#     x2 = np.random.normal(0.8,.1,n).reshape(-1,1)
    x2 = x1 + .3 + np.random.normal(0,.1,n).reshape(-1,1)
    cl = np.full(shape=(n,1), fill_value=1, dtype=int)
    d = np.hstack([x1,x2,cl])
    data = np.vstack([data, d])

    # right eye
    x1 = np.random.normal(1.3,.2,n).reshape(-1,1)
    x2 = np.random.normal(0.8,.1,n).reshape(-1,1)
    x2 = x1 - .5 + .3 + np.random.normal(0,.1,n).reshape(-1,1)
    cl = np.full(shape=(n,1), fill_value=2, dtype=int)
    d = np.hstack([x1,x2,cl])
    data = np.vstack([data, d])

    # face outline
    noise = np.random.normal(0,.1,n).reshape(-1,1)
    x1 = np.linspace(0,2,n).reshape(-1,1)
    x2 = (x1-1)**2 + noise
    cl = np.full(shape=(n,1), fill_value=3, dtype=int)
    d = np.hstack([x1,x2,cl])
    data = np.vstack([data, d])

    df = pd.DataFrame(data, columns=['x1','x2','class'])
    return df


# Test we get 4 classes in a smiley face:

# +
df = smiley(n=300)
X = df[['x1','x2']]
y = df['class']
rf = RandomForestClassifier(n_estimators=20, min_samples_leaf=30, n_jobs=-1)
rf.fit(X.values, y.values)

decision_boundaries(rf, X, y, feature_names=['x1', 'x2'], target_name='smiley', fontname="monospace",
       markers=['o','X','s','D'])
# -

# # One-dimensional classifier plots

# +
cancer = load_breast_cancer()

df = pd.DataFrame(data=cancer.data)
df.columns = [f'f{i}' for i in range(df.shape[1])]
df['y'] = cancer.target

x = df['f27']
y = df['y']

rf = RandomForestClassifier(n_estimators=10, min_samples_leaf=5)
rf.fit(x.values.reshape(-1,1), y)

decision_boundaries(rf,x,y,feature_names=['f27'], fontname="monospace", target_name='cancer', figsize=(5,1.5))
plt.tight_layout()
# -

df = smiley(n=200)
x = df[['x2']].values
y = df['class'].astype('int').values
rf = RandomForestClassifier(n_estimators=10, min_samples_leaf=10, n_jobs=-1)
rf.fit(x, y)
decision_boundaries(rf,x,y, fontname="monospace",
                    feature_names=['x2'],
                    target_name = 'smiley',
                    colors={'scatter_marker_alpha':.2},
                    figsize=(5,1.5))


