
import streamlit as st
import numpy as np
import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import base64
import dtreeviz
import matplotlib.pyplot as plt

def decisionTreeViz():
    iris = datasets.load_iris() #Irisデータを読み込む
    data, target = iris.data, iris.target #データとラベルを分ける
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0) # 学習データとテストデータへ7:3で分割

    tree = DecisionTreeClassifier() #分類問題のモデルを作成
    tree.fit(x_train, y_train) # 学習
    viz_model = dtreeviz.model(tree,
                            X_train=iris.data, y_train=iris.target,
                            feature_names=iris.feature_names,
                            class_names=[str(i) for i in iris.target_names],
                            target_name='variety')
    return viz_model

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

@make_png_animation(10, "img/wine-dtree-maxdepth-streamlit.png")
def plot_decision_boundaries(depth, X, y, feature_names=['proline', 'flavanoid'], target_name="wine"):
    t = DecisionTreeClassifier(max_depth=depth)
    t.fit(X, y)

    fig,ax = plt.subplots(1,1, figsize=(4,3.5))
    dtreeviz.decision_boundaries(t, X, y, 
           feature_names=['proline', 'flavanoid'], target_name="wine", fontname="monospace",
           ax=ax)
    plt.title(f"tree depth {depth}")
    plt.tight_layout()
    return plt

def svg_write(svg, center=True):
    """
    Disable center to left-margin align like other objects.
    """
    with open(svg, "rb") as f:
        svg_data = f.read()
    
    # Encode as base 64
    b64 = base64.b64encode(svg_data).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = f'<p style="text-align:center; display: flex; justify-content: {css_justify};">'
    html = f'{css}<img src="data:image/svg+xml;base64,{b64}"/>'

    # Write the HTML
    st.write(html, unsafe_allow_html=True)

def png_file_write(png_file_path, center=True):
    """
    Disable center to left-margin align like other objects.
    """
    with open(png_file_path, "rb") as f:
        png_data = f.read()
    
    # Encode as base 64
    b64 = base64.b64encode(png_data).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = f'<p style="text-align:center; display: flex; justify-content: {css_justify};">'
    html = f'{css}<img src="data:image/png;base64,{b64}"/>'

    # Write the HTML
    st.write(html, unsafe_allow_html=True)

viz_model = decisionTreeViz()
# v = viz_model.view(scale=1.5,fontname="monospace")
v = viz_model.view(scale=1.5,fontname="monospace", x=[1,2,3,4])
svg_write(v.save_svg())

iris = datasets.load_iris() #Irisデータを読み込む
plot_decision_boundaries(X=iris.data[:, 0:2], y=iris.target, feature_names=['proline', 'flavanoid'], target_name="wine")
png_file_write("img/wine-dtree-maxdepth-streamlit.png")

# v.save("test.svg")
# # +
# import graphviz
# from sklearn.tree import export_graphviz

# dot = export_graphviz(tree, filled=True, rounded=True, 
#                       class_names=['setosa', 'versicolor', 'virginica'],
#                       feature_names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
#                       out_file=None) 

# graph = graphviz.Source(dot) #DOT記法をレンダリング
# graph #グラフを出力
# # -

