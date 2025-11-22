import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -- Import and clean data (importing csv into pandas)
def load_data():
    # df = pd.read_csv("intro_bees.csv")
    df = pd.read_csv("https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Other/Dash_Introduction/intro_bees.csv")

    df = df.groupby(['State', 'ANSI', 'Affected by', 'Year', 'state_code'])[['Pct of Colonies Impacted']].mean()
    df.reset_index(inplace=True)
    print(df[:5])
    return df

df = load_data()
# ------------------------------------------------------------------------------

# Sector selection
sorted_year_unique = sorted(df['Year'].unique())
selected_year = st.selectbox(
    'Year', sorted_year_unique)

bee_killers = ["Disease", "Other", "Pesticides", "Pests_excl_Varroa", "Unknown", "Varroa_mites"]
selected_killers = st.selectbox(
    'Year', bee_killers)

# Filtering data
df_selected_year = df[df['Year']==selected_year]

st.write('The year chosen by user was:  ' +
         str(selected_year))

dff = df_selected_year[df_selected_year['Affected by']=='Varroa_mites']
dff_killers = df[df['Affected by']==selected_killers]

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

fig2 = px.bar(
    data_frame=dff,
    x='State',
    y='Pct of Colonies Impacted',
    hover_data=['State', 'Pct of Colonies Impacted'],
    labels={'Pct of Colonies Impacted': '% of Bee Colonies'},
    template='plotly_dark'
)
fig2.update_layout(width=1000,height=500)
st.plotly_chart(fig2)

dff_killers = dff_killers[(dff_killers["State"] == "Idaho") | (dff_killers["State"] == "New York") | (dff_killers["State"] == "New Mexico")]

fig3 = px.line(
    data_frame=dff_killers,
    x='Year',
    y='Pct of Colonies Impacted',
    color='State',
    template='plotly_dark'
)
st.plotly_chart(fig3)