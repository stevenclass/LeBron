import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Display images and layout setup
left_co, cent_co,last_co = st.columns(3)
st.title("LeBron's Game Points Prediction")
image_path = Image.open("nba-lebron-james-record-milliard-fortune-cigare.webp")
st.image(image_path,width=400)

# Sidebar for page navigation
app_page = st.sidebar.selectbox("Select Page",['Data Exploration','Visualization','Prediction'])

# Load the dataset (only once, if not already in session state)
if 'df' not in st.session_state:
    df = pd.read_csv("lebron-game-log-dataset.csv")
    st.session_state.df = df

# Use session state to store the DataFrame
df = st.session_state.df

# Data Exploration page
if app_page == 'Data Exploration':

    st.dataframe(df.head(5))
    st.subheader("01 Description of the dataset")
    st.dataframe(df.describe())

    st.subheader("02 Missing values")
    dfnull = df.isnull()/len(df)*100
    total_missing = dfnull.sum().round(2)
    st.write(total_missing)

    if total_missing[0] == 0.0:
        st.success("Congrats, you have no missing values")

    # Filter out All Star Games
    total_all_stars_games = (df['Opp'] != "@EAS")
    df = df[total_all_stars_games]
    
    st.write("Next, let's generate a new feature: Game Type (Away or Home)")
    df['Game_Type'] = df['Opp'].apply(lambda x: 'Away' if x.startswith('@') else 'Home')

    # One-hot encoding for Home/Away
    game_dummies = pd.get_dummies(df['Game_Type'], prefix='', prefix_sep='')
    df = pd.concat([df, game_dummies], axis=1)

    # Convert Min from mm:ss:00 to numerical format
    df['Min'] = df['Min'].apply(lambda x: int(x.split(':')[0]) + int(x.split(':')[1]) / 60)

    # Select only numeric columns
    df_numeric_only = df.select_dtypes(exclude=['object'])

    # Update session state with the modified DataFrame
    st.session_state.df = df_numeric_only

    st.dataframe(df_numeric_only.head(5))
    st.success("Dataset is cleaned and ready to use")

# Visualization page
if app_page == 'Visualization':

    st.subheader("03 Data Visualization")

    # Use the updated df from session state
    df = st.session_state.df
    list_columns = df.columns

    values = st.multiselect("Select two variables:", list_columns, ["FT%", "Pts"])

    # Line chart
    st.line_chart(df, x=values[0], y=values[1])

    # Bar chart
    st.bar_chart(df, x=values[0], y=values[1])

    # Pairplot
    values_pairplot = st.multiselect("Select 4 variables:", list_columns, ["Pts", "OR", "Min", "TO"])
    df2 = df[[values_pairplot[0], values_pairplot[1], values_pairplot[2], values_pairplot[3]]]
    pair = sns.pairplot(df2)
    st.pyplot(pair)

# Prediction page
if app_page == 'Prediction':

    st.title("03 Prediction")

    # Use the updated df from session state
    df = st.session_state.df
    list_columns = list(df.columns)
    list_columns.remove("Pts")

    input_lr = st.multiselect("Select variables:", list_columns, ["FGA", "OR", "TO"])

    df2 = df[input_lr]

    # Step 1: Splitting the dataset into X and y
    X = df2
    y = df["Pts"]

    # Step 2: Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Step 3: Initialize LinearRegression model
    lr = LinearRegression()

    # Step 4: Train the model
    lr.fit(X_train, y_train)

    # Step 5: Make predictions
    predictions = lr.predict(X_test)

    # Step 6: Evaluation
    mae = metrics.mean_absolute_error(predictions, y_test)
    r2 = metrics.r2_score(predictions, y_test)

    st.write("Mean Absolute Error:", mae)
    st.write("R2 output:", r2)
