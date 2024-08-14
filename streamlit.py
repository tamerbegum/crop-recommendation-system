import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(layout="wide")

@st.cache_data
def get_data():
    df = pd.read_csv("Crop_recommendation.csv")
    return df

@st.cache_data
def get_pipeline():
    pipeline = joblib.load("stream.joblib")
    return pipeline

st.title(":green[Crop] :blue[Recommendation] \U0001F33F")

main_page, data_page, model_page = st.tabs(["Main Page", "Dataset", "Model"])

# Main page

information_container = main_page.container()
information_container.subheader(":green[The Importance of Growing Crops Suitable for Soil Conditions]")
information_container.markdown("""
The agriculture sector is a fundamental carrier of the global economy, and this sector is closely related to factors such as soil quality and weather conditions. The economic importance of agriculture is quite significant. The agricultural productivity of countries not only provides food supplies to domestic markets but also creates great economic value through foreign trade. This helps many countries earn foreign exchange through the export revenues of agricultural products and contributes to national economies.

However, farmers planting fruits and vegetables suitable for the soil plays a critical role in agricultural productivity. Growing the right plant in the right soil provides higher productivity, quality, and commercial value. Moreover, these practices offer farmers a better source of income and increase the sustainability of agricultural enterprises. Therefore, farmers considering soil and climate conditions to grow suitable crops both increase their own income and secure national and global food supply. Agriculture holds great strategic importance for economic growth and food security.
""")

information_container.image("farm.png", use_column_width=True)

# Data page

df = get_data()

data_page.dataframe(df, use_container_width=True)
data_page.divider()

# Plot the count of each crop label

fig, ax = plt.subplots(figsize=(30, 5))
sns.countplot(data=df, x="label", ax=ax)
ax.set_title("Crop Counts")
data_page.pyplot(fig, clear_figure=True)

data_page.subheader(":green[**TOP 3**]")
grouped = df.groupby(by='label').mean().reset_index()

top3_most_clicked = data_page.button("TOP 3 MOST")
top3_least_clicked = data_page.button("TOP 3 LEAST")

if top3_most_clicked:
    data_page.subheader("**TOP 3 MOST**")
    for i in grouped.columns[1:]:
        data_page.subheader(f"Top 3 crops requiring the most {i}:")
        for j, k in grouped.sort_values(by=i, ascending=False)[:3][['label', i]].values:
            data_page.write(f'{j} --> {k:.2f}')

if top3_least_clicked:
    data_page.subheader("**TOP 3 LEAST**")
    for i in grouped.columns[1:]:
        data_page.subheader(f"Top 3 crops requiring the least {i}:")
        for j, k in grouped.sort_values(by=i)[:3][['label', i]].values:
            data_page.write(f'{j} --> {k:.2f}')

# Model page

pipeline = get_pipeline()

# User inputs

# st.title(":blue[Crop Recommendation System]  \U0001F33F")

model_page.image("field.png", use_column_width=True)

model_page.header(":green[About the Crop Recommendation System]")
model_page.markdown("""
This web application provides crop recommendations based on soil and climate conditions.
It uses a machine learning model trained on agricultural data to predict the most suitable crop for specific conditions.

- **N (Nitrogen):** Amount of nitrogen in the soil.
- **P (Phosphorus):** Amount of phosphorus in the soil.
- **K (Potassium):** Amount of potassium in the soil.
- **Temperature:** Average temperature in degrees Celsius.
- **Humidity:** Humidity percentage.
- **pH:** pH level of the soil.
- **Rainfall:** Amount of rainfall in millimeters.

Click the "RECOMMEND CROP!" button to get crop recommendations suitable for your conditions.
""")

col1, col2, result_col = model_page.columns([1, 1, 1])

user_N_col2 = col2.slider(label="**:green[N]**", min_value=0, max_value=150, step=1)
user_P_col2 = col2.slider(label="**:green[P]**", min_value=0, max_value=150, step=1)
user_K_col2 = col2.slider(label="**:green[K]**", min_value=0, max_value=210, step=1)
user_temperature_col1 = col1.number_input(label="**:green[Temperature]**", min_value=5.0, max_value=45.0, step=0.1)
user_humidity_col1 = col1.number_input(label="**:green[Humidity]**", min_value=10.0, max_value=100.0, step=0.01)
user_ph_col1 = col1.number_input(label="**:green[pH]**", min_value=3.0, max_value=10.0)
user_rainfall_col1 = col1.number_input(label="**:green[Rainfall]**", min_value=0.0, max_value=300.0, step=0.1)


# Prediction

user_input = pd.DataFrame({"N": user_N_col2,
                           "P": user_P_col2,
                           "K": user_K_col2,
                           "temperature": user_temperature_col1,
                           "humidity": user_humidity_col1,
                           "ph": user_ph_col1,
                           "rainfall": user_rainfall_col1}, index=[0])

result_col.dataframe(user_input)
result = pipeline.predict(user_input)

pictures = {"rice": "pics/rice.png",
            "chickpea": "pics/chickpea.png",
            "maize": "pics/maize.png",
            "pigeonpeas": "pics/pigeonpeas.png",
            "kidneybeans": "pics/kidneybeans.png",
            "mothbeans": "pics/mothbeans.png",
            "mungbean": "pics/mungbean.png",
            "blackgram": "pics/blackgram.png",
            "lentil": "pics/lentil.png",
            "pomegranate": "pics/pomegranate.png",
            "banana": "pics/banana.png",
            "mango": "pics/mango.png",
            "grapes": "pics/grapes.png",
            "watermelon": "pics/watermelon.png",
            "muskmelon": "pics/muskmelon.png",
            "apple": "pics/apple.png",
            "orange": "pics/orange.png",
            "papaya": "pics/papaya.png",
            "coconut": "pics/coconut.png",
            "cotton": "pics/cotton.png",
            "jute": "pics/jute.png",
            "coffee": "pics/coffee.png"
            }


if result_col.button("**:green[RECOMMEND CROP!]**"):
    result = pipeline.predict(user_input)[0]
    result_col.header(f"{result.upper()}!", anchor=False)
    result_col.image(pictures[result], use_column_width=True)
    st.balloons()


# To terminal  streamlit run streamlit.py
# http://localhost:8501/