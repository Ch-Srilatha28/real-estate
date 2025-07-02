import pandas as pd
import joblib
import os
from flask import Flask, request, render_template, url_for
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# Paths
DATA_PATH = "BangaloreHouseRentDtls.csv"
MODEL_PATH = "model.pkl"
ENCODERS_PATH = "encoders.pkl"

history = []

def train_model():
    df = pd.read_csv(DATA_PATH)
    df = df[['Locality', 'AvgRent', 'HouseType']].dropna()
    df.columns = ['Location', 'Rent', 'BHK']

    df = df[df['Rent'].apply(lambda x: str(x).replace(',', '').replace('.', '').isdigit())]
    df['Rent'] = df['Rent'].apply(lambda x: float(str(x).replace(',', '')))

    le_location = LabelEncoder()
    le_bhk = LabelEncoder()
    df['Location'] = le_location.fit_transform(df['Location'])
    df['BHK'] = le_bhk.fit_transform(df['BHK'])

    X = df[['Location', 'BHK']]
    y = df['Rent']

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump((le_location, le_bhk), ENCODERS_PATH)
    return model, le_location, le_bhk

# Load or train model and encoders
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
    model, le_location, le_bhk = train_model()
else:
    model = joblib.load(MODEL_PATH)
    le_location, le_bhk = joblib.load(ENCODERS_PATH)

@app.route('/')
def home():
    return render_template('client-rent_app.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    bhk = request.form['bhk']

    try:
        location_encoded = le_location.transform([location])[0]
        bhk_encoded = le_bhk.transform([bhk])[0]
    except ValueError:
        return render_template('client-rent_app.html',
                               prediction_text="Error: Location or BHK type not found in training data.",
                               history=history)

    predicted_rent = model.predict([[location_encoded, bhk_encoded]])[0]

    history.append({
        'Location': location,
        'BHK': bhk,
        'Predicted Rent': round(predicted_rent, 2)
    })

    return render_template('client-rent_app.html',
                           prediction_text=f"Estimated Rent: ₹{round(predicted_rent, 2)}",
                           history=history)

@app.route('/graph')
def graph():
    try:
        df = pd.read_csv(DATA_PATH)

        df = df[df['AvgRent'].apply(lambda x: str(x).replace(',', '').replace('.', '').isdigit())]
        df['AvgRent'] = df['AvgRent'].apply(lambda x: float(str(x).replace(',', '')))

        avg_rent = df.groupby('Locality')['AvgRent'].mean().sort_values(ascending=False).head(10)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=avg_rent.index,
                    y=avg_rent.values,
                    marker=dict(color='rgba(0,123,255,0.7)', line=dict(color='rgba(0,0,0,0.8)', width=1.5)),
                    text=[f"₹{int(val):,}" for val in avg_rent.values],
                    textposition='auto',
                    hoverinfo='x+y',
                )
            ]
        )

        fig.update_layout(
            title='🏙️ Top 10 Expensive Locations in Bangalore (Avg Rent)',
            xaxis_title='Location',
            yaxis_title='Average Rent (INR)',
            template='plotly_white',
            plot_bgcolor='#f5f5f5',
            paper_bgcolor='#f0f0f0',
            font=dict(family='Segoe UI', size=14, color='#333'),
            margin=dict(l=40, r=40, t=80, b=40)
        )

        graph_html = pio.to_html(fig, full_html=False)

        return render_template('graph.html', graph_html=graph_html)
    
    except Exception as e:
        return render_template('client-rent_app.html',
                               prediction_text=f"Error generating graph: {e}",
                               history=history)

if __name__ == "__main__":
    app.run(debug=True)
