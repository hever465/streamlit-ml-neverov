import streamlit as st
import pandas as pd
import joblib
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

st.write("""
# SimplePrediction App
""")

st.sidebar.header('Параметры')


def get_pca(current_job_yrs, experience):
    data = {
        'current_job_yrs': current_job_yrs - ((current_job_yrs+experience)/2),
        'experience': experience - ((current_job_yrs+experience)/2),
    }

    scaled_df_for_pca = pd.DataFrame(data, index=[0])
    st.write(scaled_df_for_pca)

    pca = PCA()
    df_pca = pca.fit_transform(scaled_df_for_pca)
    component_names = [f"PC{i+1}" for i in range(df_pca.shape[1])]
    df_pca = pd.DataFrame(df_pca, columns=component_names)
    st.write(df_pca)
    return df_pca


def format_func(data, option):
    return data[option]


def unclean_names(df, col):
    unclean_names = []
    for name in df[str(col)].unique():
        if name.endswith(']'):
            unclean_names.append(name)
    return unclean_names


def clean_df(df, col, unclean_list):
    for index, name in enumerate(df[col]):
        if name in unclean_list:
            if name.endswith(']'):
                name_ = name.strip('[]0123456789')
                df[col].iloc[index] = name_


def user_input_features():
    income = st.sidebar.slider('Income', 10000, 1000000, 10310)
    age = st.sidebar.slider('Возраст', 20, 80, 23)
    experience = st.sidebar.slider('Experience', 0, 20, 2)
    single = st.sidebar.selectbox('Семейное положение', options=list(SINGLE.keys()))
    car_ownership = st.sidebar.selectbox('Владеет машиной', options=list(CAR_OWNERSHIP.keys()))
    house_ownership = st.sidebar.selectbox('Квартира', options=['rented', 'owned'])
    profession = st.sidebar.selectbox('Профессия', options=list(PROFESSION.keys()))
    city = st.sidebar.selectbox('Город', options=list(CITY.keys()))
    state = st.sidebar.selectbox('Штат', options=list(STATE.keys()))
    current_job_yrs = st.sidebar.slider('Сколько лет на текущей работе', 0, 27, 12)
    current_house_yrs = st.sidebar.slider('Сколько лет на текущем месте жительства', 0, 27, 12)

    data = {
        'income': income,
        'age': age,
        'experience': experience,
        'single': SINGLE[single],
        'car_ownership': CAR_OWNERSHIP[car_ownership],
        'house_ownership': house_ownership,
        'profession': PROFESSION[profession],
        'city': CITY[city],
        'state': STATE[state],
        'current_job_yrs': current_job_yrs,
        'current_house_yrs': current_house_yrs,
        'pc1': 0.65,
        'pc2': 0.65,
    }

    features = pd.DataFrame(data, index=[0])
    df_source = pd.get_dummies(features, columns=["house_ownership"])
    # df_pca = get_pca(current_job_yrs, experience)
    st.write(df_source.shape)
    return df_source


df3 = pd.read_csv('data.csv')

features = ['Married/Single', 'Car_Ownership', 'Profession', 'CITY', 'STATE']
labels = {}

label_encoder = LabelEncoder()

for col in features:
    df3[col] = label_encoder.fit_transform(df3[col])
    if col == 'Married/Single':
        SINGLE = dict(zip( label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    if col == 'Car_Ownership':
        CAR_OWNERSHIP = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    if col == 'Profession':
        PROFESSION = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    if col == 'CITY':
        CITY = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    if col == 'STATE':
        STATE = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


df = user_input_features()

st.subheader('User Input parameters')
st.write(df)


clf = joblib.load("./random_forest.joblib")

prediction = clf.predict(df)
st.write(prediction)


# traning_data = pd.read_csv("Training Data.csv")

# clean_df(traning_data, 'STATE', unclean_names(traning_data, 'STATE'))
# clean_df(traning_data, 'CITY', unclean_names(traning_data, 'CITY'))
#
# # Метод главных компонент (PCA) для CURRENT_JOB_YRS и Experience
# features = ["CURRENT_JOB_YRS", "Experience"]
# df_for_pca = traning_data[features]
# scaled_df_for_pca = (df_for_pca - df_for_pca.mean(axis=0))/df_for_pca.std()
#
# pca = PCA()
# df_pca = pca.fit_transform(scaled_df_for_pca)
# component_names = [f"PC{i+1}" for i in range(df_pca.shape[1])]
# df_pca = pd.DataFrame(df_pca, columns=component_names)
#
# df2 = pd.concat([traning_data, df_pca], axis=1)
#
# # Балансировка данных
# class0 = df2[df2['Risk_Flag'] == 0].sample(34589)
# class1 = df2[df2['Risk_Flag'] == 1]
#
# df3 = pd.concat([class0, class1], axis=0)

# df3.to_csv('data.csv')

# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target
#
# clf = RandomForestClassifier()
# clf.fit(X, Y)


# prediction = clf.predict(df)
# prediction_proba = clf.predict_proba(df)

# st.subheader('Class labels and their corresponding index number')
# st.write(iris.target_names)
#
# st.subheader('Prediction')
# st.write(iris.target_names[prediction])
# st.write(prediction)
#
# st.subheader('Prediction Probability')
# st.write(prediction_proba)
