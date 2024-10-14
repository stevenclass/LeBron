import streamlit as st
import pandas as pd
import numpy as np
#for displaying images
from PIL import Image
import seaborn as sns
import codecs
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

left_co, cent_co,last_co = st.columns(3)

st.title("Lebron's Game Points Prediction")

image_path = Image.open("nba-lebron-james-record-milliard-fortune-cigare.webp")


st.image(image_path,width=400)


app_page = st.sidebar.selectbox("Select Page",['Data Exploration','Visualization','Prediction'])
df=pd.read_csv("lebron-game-log-dataset.csv")

if app_page == 'Data Exploration':


    st.dataframe(df.head(5))

    st.subheader("01 Description of the dataset")

    st.dataframe(df.describe())

    st.subheader("02 Missing values")

    dfnull = df.isnull()/len(df)*100
    total_missing = dfnull.sum().round(2)
    st.write(total_missing)

    if total_missing[0] == 0.0:
        st.success("Congrats you have no missing values")

    total_all_stars_games = (df['Opp'] != "@EAS")

    if total_all_stars_games.sum() != 0:
        st.write("Let's remove All Star Games since they're not a part of LeBron's season games")
    
    df = df[total_all_stars_games]
    st.write("Next, let's generate a new feature: is the Game Type - Away or Home")
    df['Game_Type'] = df['Opp'].apply(lambda x: 'Away' if x.startswith('@') else 'Home')
    
    
    st.write("Let's one hot encode it so that we can convert it to numerical columns for our prediction training model")
    game_dummies = pd.get_dummies(df['Game_Type'], prefix='', prefix_sep='')
    
    df = pd.concat([df, game_dummies], axis=1)
     # Convert to a numerical mins column
    st.write("Let's convert minutes from a mm:ss:SS format to numerical minutes out of 60")
    df['Min'] = df['Min'].apply(lambda x: int(x.split(':')[0]) + int(x.split(':')[1]) / 60)

    df_numeric_only = df.select_dtypes(exclude=['object'])

    st.dataframe(df_numeric_only.head(5))

    st.success("Now, we have a clean dataset, ready to be explored")

    if st.button("Generate Report"):

        #Function to load HTML file
        def read_html_report(file_path):
            with codecs.open(file_path,'r',encoding="utf-8") as f:
                return f.read()

        # Inputing the file path 
        html_report= read_html_report("report.html")

        # Displaying the file
        st.title("Streamlit Quality Report")
        st.components.v1.html(html_report,height=1000,scrolling=True)

if app_page == 'Visualization':
    st.subheader("03 Data Visualization")

    list_columns = df.columns

    values = st.multiselect("Select two variables:",list_columns,["Pts","OR"])

    # Creation of the line chart
    st.line_chart(df,x=values[0],y=values[1])

    # Creation of the bar chart 
    st.bar_chart(df,x=values[0],y=values[1])

    # Pairplot
    values_pairplot = st.multiselect("Select 4 variables:",list_columns,["Pts","OR","Min","TO"])

    df2 = df[[values_pairplot[0],values_pairplot[1],values_pairplot[2],values_pairplot[3]]]
    pair = sns.pairplot(df2)
    st.pyplot(pair)


if app_page == 'Prediction':

    st.title("03 Prediction")
    list_columns = df[df[""]]
    input_lr = st.multiselect("Select variables:",list_columns,["FGA","OR","TO"])

    df2 = df[input_lr]

    # Step 1 splitting the dataset into X and y
    X= df2
    # target variable
    y= df["Pts"]

    # Step 2 splitting into 4 chuncks X_train X_test y_train y_test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    # Step3 Initialize the LinearRegression
    lr = LinearRegression()

    # Step4 Train model
    lr.fit(X_train,y_train)

    #Step5 Prediction 
    predictions = lr.predict(X_test)

    #Stp6 Evaluation

    mae=metrics.mean_absolute_error(predictions,y_test)
    r2=metrics.r2_score(predictions,y_test)

    st.write("Mean Absolute Error:",mae)
    st.write("R2 output:",r2)
