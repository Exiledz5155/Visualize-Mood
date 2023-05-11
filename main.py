import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
import plotly.express as px
import glob


nltk.download('vader_lexicon')

# Filter all txt files in diary
filepaths = sorted(glob.glob("diary/*txt"))

analyzer = SentimentIntensityAnalyzer()

negativity = []
positivity = []

# Iterate through each diary and analyze the polarity, appending it to each list
for filepath in filepaths:
    with open(filepath) as file:
        content = file.read()
    scores = analyzer.polarity_scores(content)
    positivity.append(scores["pos"])
    negativity.append(scores["neg"])

# Strip the file extension and diary directory fron each filepath
dates = [name.strip(".txt").strip("diary\*") for name in filepaths]

st.title("Diary Tone")
st.subheader("Positivity")

pos_figure = px.line(x=dates, y=positivity, labels={"x": "Date", "y": "Positivity"})
st.plotly_chart(pos_figure)

st.subheader("Negativity")
neg_figure = px.line(x=dates, y=negativity, labels={"x": "Date", "y": "Negativity"})
st.plotly_chart(neg_figure)
