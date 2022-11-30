import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
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
    df[["X53416", "M83670", "X90908", "M97496"]].agg(['min','max'])

    df.y.value_counts()
    df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(columns="y")
    Y = df.y
    "### Membuang fitur yang tidak diperlukan"
    df
    X

    le = preprocessing.LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)

    "### Transformasi Label"
    Y

    le.inverse_transform(Y)

    labels = pd.get_dummies(df.y).columns.values.tolist()

    "### Label"
    labels

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    "### Normalize data transformasi"
    X

    X.shape, Y.shape

    le.inverse_transform(Y)

    labels = pd.get_dummies(df.y).columns.values.tolist()
    

with modeling:
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=4)
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

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(scoreLR))
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
    Precipitation = st.number_input('Masukkan preciptation (curah hujan) : ')
    Temp_Max = st.number_input('Masukkan tempmax (suhu maks) : ')
    Temp_Min = st.number_input('Masukkan tempmin (suhu min) : ')
    Wind = st.number_input('Masukkan wind (angin) : ')

    def submit():
        # input
        inputs = np.array([[
            Precipitation,
            Temp_Max,
            Temp_Min,
            Wind
            ]])
        le = joblib.load("le.save")
        model1 = joblib.load("knn.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang di masukkan, maka anda prediksi cuaca : {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()

