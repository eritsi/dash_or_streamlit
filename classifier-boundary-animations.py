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

# # Animations showing feature space and classification boundaries
#
# While dtreeviz is dedicated primarily to showing decision trees, we have also provided a way to show the decision boundaries for one- and two- variable classifiers. The `decision_boundaries()` function will work with any model that answers method `predict_proba()` and with Keras, for which we provided a special adapter (since that method is deprecated).
#
# Using a silly little `pltvid` library I built, we can do some simple animations.  I think it doesn't work on Windows because I directly relied on `/tmp` dir. Sorry.
#
# ## Requirements
#
# **This notebook requires poppler lib due to pltvid lib**
#
# On mac:
# ```
# brew install poppler
# ```
#
# Also needs my helper lib:

# ! pip install --quiet -U pltvid rfpimp # simple animation support by parrt

# ! sudo apt-get -y install poppler-utils

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

from rfpimp import *   # pip install rfpimp

from sklearn import tree

import dtreeviz
from dtreeviz import decision_boundaries
# -

# ## Wine data set

wine = load_wine()
X = wine.data
X = X[:,[12,6]]
y = wine.target

rf = RandomForestClassifier(n_estimators=50, min_samples_leaf=20, n_jobs=-1)
rf.fit(X, y)


# # デコレータでPNGアニメ作成部分とdecision_boundariesの描画部分を分ける

def make_png_animation(func):
    def create_canvas(*args, **kwargs):
        import pltvid
        
        dpi = 300
        camera = pltvid.Capture(dpi=dpi)
        max = 10
        
        for depth in range(1,max+1):
            func(depth, *args, **kwargs)
            if depth>=max:
                camera.snap(8)
            else:
                camera.snap()
        camera.save("img/wine-dtree-maxdepth-decorator.png", duration=500) # animated png
    return create_canvas


@make_png_animation
def plot_decision_boundaries(depth, X, y, feature_names=['proline', 'flavanoid'], target_name="wine"):
    t = DecisionTreeClassifier(max_depth=depth)
    t.fit(X, y)

    fig,ax = plt.subplots(1,1, figsize=(4,3.5))
    decision_boundaries(t, X, y, 
           feature_names=feature_names, target_name=target_name, fontname="monospace",
           ax=ax)
    plt.title(f"tree depth {depth}")
    plt.tight_layout() # この次の行で先に　plt.show()してしまうとcameraの方で白紙アニメになってしまう
    return plt


plot_decision_boundaries(X, y, feature_names=['proline', 'flavanoid'], target_name="wine")


# # さらに、デコレータに引数を持たせるケース

def make_png_animation(max_depth, png_name): # デコレータに引数を与えたい場合はネスト階層を一つ深くする必要がある
    # https://zenn.dev/umeko/articles/8ef2df8be8b017#%E3%82%AF%E3%83%AD%E3%83%BC%E3%82%B8%E3%83%A3%E3%82%92%E6%B4%BB%E7%94%A8%E3%81%97%E3%81%A6%E5%BC%95%E6%95%B0%E3%82%92%E6%B8%A1%E3%81%99
    def make_png_animation_wrapper(func):
        def create_canvas(*args, **kwargs):
            import pltvid
            
            dpi = 300
            camera = pltvid.Capture(dpi=dpi)
            max = max_depth
            
            for depth in range(1,max+1):
                func(depth, *args, **kwargs)
                if depth>=max:
                    camera.snap(8)
                else:
                    camera.snap()
            camera.save(png_name, duration=500) # animated png
        return create_canvas
    return make_png_animation_wrapper


@make_png_animation(10, "img/wine-dtree-maxdepth-decorator-wrapper.png")
def plot_decision_boundaries(depth, X, y, feature_names=['proline', 'flavanoid'], target_name="wine"):
    t = DecisionTreeClassifier(max_depth=depth)
    t.fit(X, y)

    fig,ax = plt.subplots(1,1, figsize=(4,3.5))
    decision_boundaries(t, X, y, 
           feature_names=['proline', 'flavanoid'], target_name="wine", fontname="monospace",
           ax=ax)
    plt.title(f"tree depth {depth}")
    plt.tight_layout()
    return plt


plot_decision_boundaries(X, y, feature_names=['proline', 'flavanoid'], target_name="wine")

# # 以下、dtreeviz githubの元コード

# +
import pltvid

dpi = 300
camera = pltvid.Capture(dpi=dpi)
max = 10
for depth in range(1,max+1):
    t = DecisionTreeClassifier(max_depth=depth)
    t.fit(X, y)

    fig,ax = plt.subplots(1,1, figsize=(4,3.5), dpi=dpi)
    decision_boundaries(t, X, y, 
           feature_names=['proline', 'flavanoid'], target_name="wine", fontname="monospace",
           ax=ax)
    plt.title(f"Wine tree depth {depth}")
    plt.tight_layout()
    if depth>=max:
        camera.snap(8)
    else:
        camera.snap()
    # plt.show()

camera.save("wine-dtree-maxdepth.png", duration=500) # animated png


# -

# ## Synthetic data set

def smiley(n = 1000):
    # mouth
    x1 = np.random.normal(1.0,.2,n).reshape(-1,1)
    x2 = np.random.normal(0.4,.05,n).reshape(-1,1)
    cl = np.full(shape=(n,1), fill_value=0, dtype=int)
    d = np.hstack([x1,x2,cl])
    data = d
    
    # left eye
    x1 = np.random.normal(.7,.2,n).reshape(-1,1)
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


# ### Animate num trees in RF

# +
import pltvid

df = smiley(n=100)
X = df[['x1','x2']]
y = df['class']
rf = RandomForestClassifier(n_estimators=10, min_samples_leaf=1, n_jobs=-1)
rf.fit(X, y)

dpi = 300
camera = pltvid.Capture(dpi=dpi)
max = 100
tree_sizes = [*range(1,10)]+[*range(10,max+1,5)]
for nt in tree_sizes:
    np.random.seed(1) # use same bagging sets for animation
    rf = RandomForestClassifier(n_estimators=nt, min_samples_leaf=1, n_jobs=-1)
    rf.fit(X.values, y.values)

    fig,ax = plt.subplots(1,1, figsize=(5,3.5), dpi=dpi)
    decision_boundaries(rf, X.values, y, feature_names=['x1', 'x2'], fontname="monospace",
                 ntiles=70, dot_w=15, boundary_markersize=.4, ax=ax)
    plt.title(f"Synthetic dataset, {nt} trees")
    plt.tight_layout()
    if nt>=tree_sizes[-1]:
        camera.snap(5)
    else:
        camera.snap()
    # plt.show()

camera.save("smiley-numtrees.png", duration=500)
# -

# ### Animate decision tree max depth

# +
import pltvid

df = smiley(n=100) # more stark changes with fewer
X = df[['x1','x2']]
y = df['class']

dpi = 300
camera = pltvid.Capture(dpi=dpi)
max = 10
for depth in range(1,max+1):
    t = DecisionTreeClassifier(max_depth=depth)
    t.fit(X.values, y.values)

    fig,ax = plt.subplots(1,1, figsize=(5,3.5), dpi=dpi)
    decision_boundaries(t, X, y, 
               feature_names=['x1', 'x2'], target_name="class", fontname="monospace",
               colors={'scatter_edge': 'black',
                       'tessellation_alpha':.6},
               ax=ax)
    plt.title(f"Synthetic dataset, tree depth {depth}")
    plt.tight_layout()
    if depth>=max:
        camera.snap(8)
    else:
        camera.snap()
    # plt.show()

camera.save("smiley-dtree-maxdepth.png", duration=500)
# -

# ### Animate decision tree min samples per leaf

# +
import pltvid

df = smiley(n=100)
X = df[['x1','x2']]
y = df['class']

dpi = 300
camera = pltvid.Capture(dpi=dpi)
max = 20
for leafsz in range(1,max+1):
    t = DecisionTreeClassifier(min_samples_leaf=leafsz)
    t.fit(X.values, y.values)

    fig,ax = plt.subplots(1,1, figsize=(5,3.5), dpi=dpi)
    decision_boundaries(t, X, y, 
               feature_names=['x1', 'x2'], target_name="class", fontname="monospace",
               colors={'scatter_edge': 'black',
                       'tessellation_alpha':.4},
               ax=ax)
    plt.title(f"Synthetic dataset, {leafsz} samples/leaf")
    plt.tight_layout()
    if leafsz>=max:
        camera.snap(8)
    else:
        camera.snap()
    # plt.show()

camera.save("smiley-dtree-minsamplesleaf.png", duration=500)
# -


