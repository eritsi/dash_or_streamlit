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

# # 自作の関数で最後のリーフごとに追加のsvg描画を行う

# +
from dtreeviz.trees import _regr_leaf_viz, adjust_colors
from dtreeviz.utils import _format_axes
import matplotlib.pyplot as plt

def _my_regr_leaf_viz(df,
                   y: (pd.Series, np.ndarray),
                   target_name: str,
                   filename: str,
                   label_fontsize: int,
                   ticks_fontsize: int,
                   fontname: str,
                   colors):
    colors = adjust_colors(colors)

    # samples = node.samples()
    samples = len(df)
    # y = y[samples]

    figsize = (.75, .8)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # m = node.prediction()

    _format_axes(ax, None, None, colors, fontsize=label_fontsize, fontname=fontname, ticks_fontsize=ticks_fontsize, grid=False)
    # ax.set_ylim(y_range)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])

    ticklabelpad = plt.rcParams['xtick.major.pad']
    ax.annotate(f"{target_name}\nn={len(y)}",
                xy=(.5, 0), xytext=(.5, -.5 * ticklabelpad), ha='center', va='top',
                xycoords='axes fraction', textcoords='offset points',
                fontsize=label_fontsize, fontname=fontname, color=colors['axis_label'])

    mu = .5
    sigma = .08
    X = np.random.normal(mu, sigma, size=len(y)) # !!!適当にX方向に広げてる
    ax.set_xlim(0, 1)
    alpha = colors['scatter_marker_alpha']  # was .25

    ax.scatter(X, y, s=5, c=colors['scatter_marker'], alpha=alpha, lw=.3)
    # ax.plot([0, len(node.samples())], [m, m], '--', color=colors['split_line'], linewidth=1) # 平均の点線は不要
    
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

dtreeviz.trees._regr_leaf_viz = _my_regr_leaf_viz

# +
from dtreeviz.trees import _draw_piechart, adjust_colors
from dtreeviz.models.shadow_decision_tree import ShadowDecTreeNode
from typing import List
def _my_class_leaf_viz(node: ShadowDecTreeNode,
                    colors: List[str],
                    filename: str,
                    graph_colors,
                    fontname,
                    leaftype):
    graph_colors = adjust_colors(graph_colors)

    minsize = .15
    maxsize = 1.3
    slope = 0.02
    nsamples = node.nsamples()
    size = nsamples * slope + minsize
    size = min(size, maxsize)

    # we visually need n=1 and n=9 to appear different but diff between 300 and 400 is no big deal
    counts = node.class_counts()
    prediction = node.prediction_name()

    # when using another dataset than the training dataset, some leaves could have 0 samples.
    # Trying to make a pie chart will raise some deprecation
    if sum(counts) == 0:
        return
    if leaftype == 'pie':
        _draw_piechart(counts, size=size, colors=colors, filename=filename, label=f"n={nsamples}\n{prediction}\nleaf#{node.id}",
                      graph_colors=graph_colors, fontname=fontname)
        _my_regr_leaf_viz(df=pd.DataFrame(x_train)[0:10],y=pd.DataFrame(x_train).iloc[0:10, 0:1],target_name='AI_score',
                      filename=filename.split('.')[0]+'_score.svg', label_fontsize=12, ticks_fontsize=8, fontname=fontname, colors=graph_colors)
    elif leaftype == 'barh':
        _draw_barh_chart(counts, size=size, colors=colors, filename=filename, label=f"n={nsamples}\n{prediction}",
                      graph_colors=graph_colors, fontname=fontname)
    else:
        raise ValueError(f'Undefined leaftype = {leaftype}')

dtreeviz.trees._class_leaf_viz = _my_class_leaf_viz
# -

# 変化は少ない（リーフ番号を表示しただけ）が、　_score.svgが作られている
viz_model = dtreeviz.model(tree,
                               X_train=iris.data, y_train=iris.target,
                               feature_names=iris.feature_names,
                               class_names=[str(i) for i in iris.target_names],
                               target_name='variety')
v = viz_model.view(scale=1.5, fontname="monospace")
v.save("test3.svg")
v

# # .dotファイルを編集し、追加の描画を図に挿入する

# +
import re

def edit_dot_file(input_path, output_path):
    # Read the contents of the input .dot file
    with open(input_path, 'r') as file:
        content = file.read()
    
    # Define the regex pattern to match the target block
    pattern = re.compile(r'(<tr>\s*<td><img src="/tmp/leaf\d+_\d+\.svg"/></td>\s*</tr>)')
    
    # Define the replacement function
    def replacement(match):
        original_block = match.group(1)
        new_block = original_block + '\n                <tr>\n                        <td><img src="/tmp/leaf{}_{}_score.svg"/></td>\n                </tr>'.format(*re.findall(r'leaf(\d+)_(\d+)', original_block)[0])
        return new_block
    
    # Replace the matched blocks with the modified content
    new_content = re.sub(pattern, replacement, content)
    
    # Write the new content to the output .dot file
    with open(output_path, 'w') as file:
        file.write(new_content)

# Example usage
edit_dot_file('/workspace/dash_or_streamlit/test3', '/workspace/dash_or_streamlit/test3-score')
# -

from dtreeviz.utils import DTreeVizRender
with open('/workspace/dash_or_streamlit/test3-score', 'r') as file:
        content = file.read()
vi =DTreeVizRender(content, 1.0)
vi

vi.save('test3-score.svg')

# # 単純に並べたい場合は簡易な.dotファイルを作成する

from dtreeviz.utils import DTreeVizRender
dot_score = f"""
digraph G {{
    splines=line;
    
    leaf1 [margin="0" shape=box penwidth="0" color="#444443" label=<<table border="0" CELLBORDER="0">
                
                <tr>
                        <td><img src="/tmp/leaf1_6553.svg"/></td>
                </tr>
                <tr>
                        <td><img src="/tmp/leaf1_6553_score.svg"/></td>
                </tr>
                </table>>]
	leaf4 [margin="0" shape=box penwidth="0" color="#444443" label=<<table border="0" CELLBORDER="0">
                
                <tr>
                        <td><img src="/tmp/leaf4_6553.svg"/></td>
                </tr>
                <tr>
                        <td><img src="/tmp/leaf4_6553_score.svg"/></td>
                </tr>
                </table>>]
	leaf6 [margin="0" shape=box penwidth="0" color="#444443" label=<<table border="0" CELLBORDER="0">
                
                <tr>
                        <td><img src="/tmp/leaf6_6553.svg"/></td>
                </tr>
                <tr>
                        <td><img src="/tmp/leaf6_6553_score.svg"/></td>
                </tr>
                </table>>]
	leaf7 [margin="0" shape=box penwidth="0" color="#444443" label=<<table border="0" CELLBORDER="0">
                
                <tr>
                        <td><img src="/tmp/leaf7_6553.svg"/></td>
                </tr>
                <tr>
                        <td><img src="/tmp/leaf7_6553_score.svg"/></td>
                </tr>
                </table>>]
	leaf10 [margin="0" shape=box penwidth="0" color="#444443" label=<<table border="0" CELLBORDER="0">
                
                <tr>
                        <td><img src="/tmp/leaf10_6553.svg"/></td>
                </tr>
                <tr>
                        <td><img src="/tmp/leaf10_6553_score.svg"/></td>
                </tr>
                </table>>]
	leaf11 [margin="0" shape=box penwidth="0" color="#444443" label=<<table border="0" CELLBORDER="0">
                
                <tr>
                        <td><img src="/tmp/leaf11_6553.svg"/></td>
                </tr>
                <tr>
                        <td><img src="/tmp/leaf11_6553_score.svg"/></td>
                </tr>
                </table>>]
	leaf12 [margin="0" shape=box penwidth="0" color="#444443" label=<<table border="0" CELLBORDER="0">
                
                <tr>
                        <td><img src="/tmp/leaf12_6553.svg"/></td>
                </tr>
                <tr>
                        <td><img src="/tmp/leaf12_6553_score.svg"/></td>
                </tr>
                </table>>]
}}
    """
DTreeVizRender(dot_score, 1.0)

from dtreeviz.trees import _draw_piechart, adjust_colors
import matplotlib.pyplot as plt
def _my_draw_piechart(counts, size, colors, filename, label, fontname, graph_colors):
    graph_colors = adjust_colors(graph_colors)
    n_nonzero = np.count_nonzero(counts)

    if n_nonzero != 0:
        i = np.nonzero(counts)[0][0]
        if n_nonzero == 1:
            counts = [counts[i]]
            colors = [colors[i]]

    tweak = size * .01
    fig, ax = plt.subplots(1, 1, figsize=(size, size))
    ax.axis('equal')
    ax.set_xlim(0, size - 10 * tweak)
    ax.set_ylim(0, size - 10 * tweak)
    # frame=True needed for some reason to fit pie properly (ugh)
    # had to tweak the crap out of this to get tight box around piechart :(
    wedges, _ = ax.pie(counts, center=(size / 2 - 6 * tweak, size / 2 - 6 * tweak), radius=size / 2, colors=colors,
                       shadow=False, frame=True)
    for w in wedges:
        w.set_linewidth(.5)
        w.set_edgecolor(graph_colors['pie'])

    ax.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if label is not None:
        ax.text(size / 2 - 6 * tweak, -10 * tweak, label,
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=9, color=graph_colors['text'], fontname=fontname)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
dtreeviz.trees._draw_piechart = _my_draw_piechart

viz_model.ctree_leaf_distributions(fontname="monospace")

viz_model.rtree_leaf_distributions(fontname="monospace")

# Xが1または2列（特徴量の数）である必要あり
tree = DecisionTreeClassifier() #分類問題のモデルを作成
tree.fit(iris.data[:, 0:2], iris.target)
dtreeviz.decision_boundaries(tree, X=iris.data[:, 0:2], y=iris.target, fontname="monospace",
       feature_names=iris.feature_names[0:2])


v.svg()


