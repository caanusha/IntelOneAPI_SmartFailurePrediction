from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import datetime
import numpy as np  # np mean, np random
import plotly.express as px  # interactive charts
import streamlit as st
import serial
import pandas as pd
from PIL import Image
import pickle

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

icon = Image.open("icon.ico")
st.set_page_config(
    page_title="Real-Time Device Data Collector Dashboard📈",
    page_icon=icon,
    layout="wide",
)
is_collecting_data = False

# Define a list of options for the dropdown
options = ["Two-Wheeler Engine", "Four-Wheeler Engine", "Mixer-Grinder"]

# Define a list of options for the dropdown
algoOptions = ["KNN", "Decision Tree", "Random-Forest", "Support Vector Machine", "Logistic Regression"]


# dashboard title
st.title("Live Device Data Collector Dashboard📈")

# creating a single-element container
placeholder = st.empty()

df = pd.DataFrame(
    {'Time': ['2023-03-25 18:58:32'], 'Sound': [30], 'Vibration': [100], 'Temperature': [30], 'Humidity': [30]})


def startDataCollection():
    global is_collecting_data
    is_collecting_data = True

    try:
        ser = serial.Serial('COM6', 9600)
        ser.flushInput()
        fields = ['Time', 'Sound', 'Vibration', 'Temperature', 'Humidity']
        # near real-time / live feed simulation
        while is_collecting_data:
            # Read data from the serial port
            s = ser.readline().decode()
            # Convert data to float and append to the data list
            try:
                rows = [float(x) for x in s.split(',')]
                timeline = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                rows.insert(0, timeline)
                if len(rows) == 5:
                    # Convert str to float data type
                    timevalue = np.array(rows[0])
                    sound = np.array(float(rows[1]))
                    vibration = np.array(float(rows[2]))
                    temperature = np.array(float(rows[3]))
                    humidity = np.array(float(rows[4]))

                    # add the new row to the DataFrame using loc accessor
                    df.loc[len(df)] = [timevalue, sound, vibration, temperature, humidity]

                    # saving the dataframe
                    df.to_csv('RealTimeData.csv')

                    data_count = int(
                        df["Sound"].count()
                    )

                    # creating KPIs
                    avg_sound = np.mean(df["Sound"])
                    avg_vibration = np.mean(df["Vibration"])
                    avg_temperature = np.mean(df["Temperature"])
                    avg_humidity = np.mean(df["Humidity"])

                    with placeholder.container():
                        # create five columns
                        dataCount, avgSound, avgVibration, avgTemperature, avgHumidity = st.columns(5)

                        # fill in those three columns with respective metrics or KPIs
                        dataCount.metric(
                            label="Count 🧮",
                            value=data_count,
                        )
                        # fill in those three columns with respective metrics or KPIs
                        avgSound.metric(
                            label="Avg(Sound) 🖩",
                            value=round(avg_sound),
                        )
                        # fill in those three columns with respective metrics or KPIs
                        avgVibration.metric(
                            label="Avg(Vibration) 🖩",
                            value=round(avg_vibration),
                        )
                        # fill in those three columns with respective metrics or KPIs
                        avgTemperature.metric(
                            label="Avg(Temperature) 🖩",
                            value=round(avg_temperature),
                        )
                        # fill in those three columns with respective metrics or KPIs
                        avgHumidity.metric(
                            label="Avg(Humidity) 🖩",
                            value=round(avg_humidity),
                        )

                        # create two columns for charts
                        fig_sound, fig_vibration = st.columns(2)
                        with fig_sound:
                            st.markdown("### Sound📢 VS Time⏱️")
                            figSound = px.line(
                                data_frame=df, y="Sound", x="Time"
                            )
                            st.write(figSound)

                        with fig_vibration:
                            st.markdown("### Vibration📳 VS Time⏱️")
                            figVibration = px.line(data_frame=df, y="Vibration", x="Time")
                            st.write(figVibration)

                        # create two columns for charts
                        fig_temperature, fig_humidity = st.columns(2)
                        with fig_temperature:
                            st.markdown("### Temperature🌡️ VS Time⏱️")
                            figTemperature = px.line(
                                data_frame=df, y="Temperature", x="Time"
                            )
                            st.write(figTemperature)

                        with fig_humidity:
                            st.markdown("### Humidity☁️ VS Time⏱️")
                            figHumidity = px.line(data_frame=df, y="Humidity", x="Time")
                            st.write(figHumidity)

                        st.markdown("### Detailed Data View")
                        st.dataframe(df)
            except:
                pass
    except:
        print("Connection Unsuccesful!")

# Define a function that takes a data frame and makes predictions
def predictusing(selectedAlgo):
    # Load the pre-trained model from disk
    if selectedAlgo == algoOptions[0]:
        with open('../Training/Models/KNNmodel.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selectedAlgo == algoOptions[1]:
        with open('../Training/Models/DTmodel.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selectedAlgo == algoOptions[2]:
        with open('../Training/Models/RFmodel.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selectedAlgo == algoOptions[3]:
        with open('../Training/Models/SVMmodel.pkl', 'rb') as f:
            model = pickle.load(f)
    elif selectedAlgo == algoOptions[4]:
        with open('../Training/Models/LRmodel.pkl', 'rb') as f:
            model = pickle.load(f)
        # Make predictions using the pre-trained model

    df = pd.read_csv('RealTimeData.csv')
    print(df)
    temp = df.drop(df.columns[[0, 1]], axis=1)
    means = temp.mean()
    print(means)
    means_2d = means.values.reshape(1, -1)
    predictions = model.predict(means_2d)
    return predictions


# Define function to stop data collection
def compute(selectedAlgo):
    global is_collecting_data
    is_collecting_data = False
    # Call the predict function with the new data frame
    prediction = predictusing(selectedAlgo)

    # Print the prediction
    print(prediction)
    return prediction


def main():
    with st.sidebar:
        st.markdown("# Real Time Data Uploader")
        # Add a dropdown with a default value to the app
        selected_option = st.selectbox("Select the device", options, index=0)

        # Add a button to the app
        if st.button("Start Data Collection", key="start"):
            startDataCollection()

        if st.button("Stop Data Collection", key="stop"):
            st.stop()

        # Add a dropdown with a default value to the app
        selectedAlgo = st.selectbox("Algorithm", algoOptions, index=0)

        if st.button("Compute", key="compute"):
            results = compute(selectedAlgo)
            # Create a textarea
            if results == 1:
                results_area = st.text_area('Results:', "Good")
            else:
                results_area = st.text_area('Results:', "Bad!!! Maintenance Needed!!!")


main()