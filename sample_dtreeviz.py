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

# +
import numpy as np
import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split


iris = datasets.load_iris() #Irisデータを読み込む
data, target = iris.data, iris.target #データとラベルを分ける
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0) # 学習データとテストデータへ7:3で分割

print(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype) #データ型の確認
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) #データ数の確認


# +
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier() #分類問題のモデルを作成
tree.fit(x_train, y_train) # 学習
y_pred = tree.predict(x_test) # テストデータの予測値

print(tree.get_params())
print(y_pred)
print('学習時スコア：', tree.score(x_train, y_train), '検証スコア', tree.score(x_test, y_test))


# +
import graphviz
from sklearn.tree import export_graphviz

dot = export_graphviz(tree, filled=True, rounded=True, 
                      class_names=['setosa', 'versicolor', 'virginica'],
                      feature_names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
                      out_file=None) 

graph = graphviz.Source(dot) #DOT記法をレンダリング
graph #グラフを出力
# -

import dtreeviz
viz_model = dtreeviz.model(tree,
                           X_train=iris.data, y_train=iris.target,
                           feature_names=iris.feature_names,
                           class_names=[str(i) for i in iris.target_names],
                           target_name='variety')
v = viz_model.view(scale=1.5,fontname="monospace")
v.save("test2.svg")
v


v = viz_model.view(scale=1.5,fontname="monospace",x=[1,2,3,4])
v

viz_model.ctree_leaf_distributions(fontname="monospace")

viz_model.rtree_leaf_distributions(fontname="monospace")

# Xが1または2列（特徴量の数）である必要あり
tree = DecisionTreeClassifier() #分類問題のモデルを作成
tree.fit(iris.data[:, 0:2], iris.target)
dtreeviz.decision_boundaries(tree, X=iris.data[:, 0:2], y=iris.target, fontname="monospace",
       feature_names=iris.feature_names[0:2])


v.svg()


