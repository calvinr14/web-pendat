import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib

st.title("PENAMBANGAN DATA")
st.write("##### Nama  : Calvin Rifansyah")
st.write("##### Nim   : 200411100072 ")
st.write("##### Kelas : Penambangan Data C ")
data_set_description, upload_data, preporcessing, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Prepocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : Brain Tumor (Tumor Otak) ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/jillanisofttech/brain-tumor")
    
with upload_data:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with preporcessing:
    st.write("""# Preprocessing""")
    df[["Unnamed: 0", "X53416", "M83670", "X90908"]].agg(['min','max'])

    df.y.value_counts()
    df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(columns="y")
    Y = df.brain

    "### Penghapusan Fitur"
    df
    X

    le = preprocessing.LabelEncoder()
    le.fit(y)
    Y = le.transform(y)


    le = LabelEncoder()
    Y = le.fit_transform(y)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    st.write("Hasil Preprocesing : ", scaled)

    "### Transformasi Label"
    y

    le.inverse_transform(y)

    labels = pd.get_dummies(df.brain).columns.values.tolist()

    "### Label"
    labels

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    "### Normalize data transformasi"
    X

    X.shape, y.shape

    le.inverse_transform(y)

    labels = pd.get_dummies(df.brain).columns.values.tolist()
    
    "### Label"
    labels

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X

    X.shape, y.shape
    
with modeling:
    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    lr = st.checkbox('LogisticRegression')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")


    # Fit LogisticRegression
    log_reg = LogisticRegression()
    log_reg.fit(X_train,Y_train)

    # Predicting the Test set results
    y_predict = log_reg.predict(X_test)
    
   ## Accuracy
    scoreLR = log_reg.score(X_test, Y_test)

    # KNN 
    KNN = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
    KNN.fit(X_train,Y_train)

    # Prediction
    y_predict = KNN.predict(X_test)
    
    # Accuracy Score
    scoreKNN = KNN.score(X_test, Y_test)

    # RandomForestClassifier

    # Fit DecisionTree Classifier
    DTC = DecisionTreeClassifier()
    DTC.fit(X_train,Y_train)

    #prediction
    y_predict = DTC.predict(X_test)

    #Accuracy
    scoredt = DTC.score(X_test, Y_test)

    if lr :
        if mod :
            st.write('Model Logistic Regression accuracy score: {0:0.2f}'. format(scoreLR))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(scoreKNN))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(scoredt))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [scoreLR,scoreKNN,scoredt],
            'Nama Model' : ['Logistic Regression','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            Y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True) 

with implementation:
    st.write("# Implementation")
    X53416 = st.number_input('Input X53416 : ')
    M83670 = st.number_input('Input M83670 : ')
    X909081 = st.number_input('Input X90908 : ')
    M97496 = st.number_input('Input M97496 : ')
    X90908 = st.number_input('Input X90908.1 : ')
    U37019 = st.number_input('Input U37019 : ')
    R48602 = st.number_input('Input R48602 : ')
    T96548 = st.number_input('Input T96548 : ')

    def submit():
        # input
        inputs = np.array([[
            X53416,
            M83670,
            X909081,
            M97496,
            X90908,
            U37019,
            R48602,
            T96548
            ]])

        le = joblib.load("le.save")

        if scoreLR > scoreKNN and scoredt:
            model1 = joblib.load("lr.joblib")

        elif scoreKNN > scoreLR and scoredt:
            model1 = joblib.load("knn.joblib")

        elif scoredt > scoreKNN and scoreLR:
            model1 = joblib.load("dtc.joblib")

        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang di masukkan, maka pasien termasuk : {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()

