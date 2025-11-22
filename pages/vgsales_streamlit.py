import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -- Import and clean data (importing csv into pandas)
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Other/Excel_Dash/vgsales.csv")

    print(df[:5])
    return df

df = load_data()
# ------------------------------------------------------------------------------

# Sector selection
sorted_genre_unique = sorted(df['Genre'].unique())
selected_genre = st.selectbox(
    'Genre', sorted_genre_unique)

sorted_platform_unique = sorted(df['Platform'].unique())
selected_platform = st.selectbox(
    'Platform', sorted_platform_unique)

# Filtering data
dff = df[(df['Genre']==selected_genre) & (df['Platform']==selected_platform)].sort_values('Japan Sales', ascending=False)
dfff = pd.melt(dff,id_vars=['Rank','Video Game','Platform','Year','Genre','Publisher'],var_name='Country',value_name='Sales')

fig = px.bar(data_frame=dff, x='Year', y='Japan Sales', hover_name='Video Game')
st.plotly_chart(fig)
fig2 = px.bar(data_frame=dfff, x='Year', y='Sales', color='Country', hover_name='Video Game', barmode='relative')
st.plotly_chart(fig2)

st.write(dff[dff['Year']==2000])
st.write(dfff)

fig_pie = px.pie(data_frame=df, names='Genre', values='Japan Sales')
st.plotly_chart(fig_pie)