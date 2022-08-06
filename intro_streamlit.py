import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -- Import and clean data (importing csv into pandas)
@st.cache
def load_data():
    # df = pd.read_csv("intro_bees.csv")
    df = pd.read_csv("https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Other/Dash_Introduction/intro_bees.csv")

    df = df.groupby(['State', 'ANSI', 'Affected by', 'Year', 'state_code'])[['Pct of Colonies Impacted']].mean()
    df.reset_index(inplace=True)
    print(df[:5])
    return df

df = load_data()
# ------------------------------------------------------------------------------

# Sidebar - Sector selection
sorted_year_unique = sorted(df['Year'].unique())
selected_year = st.selectbox(
    'Year', sorted_year_unique)

# Filtering data
df_selected_year = df[df['Year']==selected_year]

st.write('The year chosen by user was:  ' +
         str(selected_year))

dff = df_selected_year[df_selected_year['Affected by']=='Varroa_mites']

fig = px.choropleth(
        data_frame=dff,
        locationmode='USA-states',
        locations='state_code',
        scope="usa",
        color='Pct of Colonies Impacted',
        hover_data=['State', 'Pct of Colonies Impacted'],
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={'Pct of Colonies Impacted': '% of Bee Colonies'},
        template='plotly_dark'
    )

st.plotly_chart(fig)