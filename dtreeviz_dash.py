from dash import Dash, html
import numpy as np
import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import base64
import dtreeviz

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

#################   streamlit  x  SVG
# def svg_write(svg, center=True):
#     """
#     Disable center to left-margin align like other objects.
#     """
#     with open(svg, "rb") as f:
#         svg_data = f.read()
    
#     # Encode as base 64
#     b64 = base64.b64encode(svg_data).decode("utf-8")

#     # Add some CSS on top
#     css_justify = "center" if center else "left"
#     css = f'<p style="text-align:center; display: flex; justify-content: {css_justify};">'
#     html = f'{css}<img src="data:image/svg+xml;base64,{b64}"/>'

#     # Write the HTML
#     st.write(html, unsafe_allow_html=True)

#################   dash  x  SVG
def svg_write(svg):
    """
    Disable center to left-margin align like other objects.
    """
    with open(svg, "rb") as f:
        svg_data = f.read()
    
    # Encode as base 64
    b64 = base64.b64encode(svg_data).decode("utf-8")

    # Write the HTML, cf. PNG   html.Img(src='data:image/png;base64,{}'.format(b64))
    return html.Img(src='data:image/svg+xml;base64,{}'.format(b64))

viz_model = decisionTreeViz()
# v = viz_model.view(scale=1.5,fontname="monospace")
v = viz_model.view(scale=1.5,fontname="monospace", x=[1,2,3,4])

app = Dash()

app.layout = html.Div([
          svg_write(v.save_svg())
    ])

if __name__ == '__main__':
    app.run_server(debug=True)