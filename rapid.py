import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ta
import streamlit as st
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from newsapi import NewsApiClient

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Yahoo Finance RapidAPI configuration
RAPIDAPI_KEY = "a85439da56msh17f1223a17bbfc6p10807ejsnf3da307dc41b"
YAHOO_FINANCE_URL = "https://yahoo-finance160.p.rapidapi.com/stock/v3/get-historical-data"

# Initialize NewsAPI client
NEWS_API_KEY = "71e6d08d6731464e80b52967917470b0"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# =========================
# Helper Functions (Fixed)
# =========================

def get_stock_first_date(ticker):
    """Get approximate start date for stocks"""
    return datetime(2010, 1, 1).date()

def get_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance RapidAPI"""
    try:
        # Convert dates to timestamp format
        start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "yahoo-finance160.p.rapidapi.com"
        }

        params = {
            "symbol": ticker,
            "period1": start_timestamp,
            "period2": end_timestamp
        }

        response = requests.get(YAHOO_FINANCE_URL, headers=headers, params=params)
        data = response.json()

        # Check if valid data exists
        if 'prices' not in data or not isinstance(data['prices'], list):
            st.warning(f"No data available for {ticker} between {start_date} and {end_date}")
            return pd.DataFrame()

        # Process the API response
        df = pd.DataFrame(data['prices'])
        
        # Check if dataframe is empty
        if df.empty:
            st.warning("Received empty data from API")
            return pd.DataFrame()

        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['date'], unit='s')
        
        # Filter and rename columns
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Filter out weekends and invalid rows
        df = df[df['Date'].dt.dayofweek < 5]
        df = df[df['Close'] > 0]  # Remove invalid price rows
        
        # Sort by date ascending
        df = df.sort_values('Date', ascending=True)

        # Add technical indicators
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['7_day_avg'] = df['Close'].rolling(window=7).mean()
        df['30_day_avg'] = df['Close'].rolling(window=30).mean()
        
        # Drop rows with missing values
        df.dropna(inplace=True)

        return df

    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return pd.DataFrame()

def fetch_news_sentiment(ticker, api_key):
    """Fetch news sentiment using NewsAPI"""
    try:
        articles = newsapi.get_everything(q=ticker.split('.')[0],
                                        language='en',
                                        sort_by='relevancy',
                                        page_size=20)
        if articles['totalResults'] == 0:
            return 0

        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(article['title'])['compound'] 
                for article in articles['articles']]
        return np.mean(scores) if scores else 0
    except:
        return 0

# =========================
# Linear Regression Functions
# =========================

def train_and_evaluate_model(df):
    """Train and evaluate Linear Regression model"""
    df['Date_Ordinal'] = df['Date'].map(lambda x: x.toordinal())
    X = df[['Date_Ordinal', 'Open', 'High', 'Low', 'Volume', 
           'RSI', 'BB_High', 'BB_Low', '7_day_avg', '30_day_avg', 'Sentiment']]
    y = df['Close']
    
    X.columns = ['Date_Ordinal', 'Open', 'High', 'Low', 'Volume', 
                'RSI', 'BB_High', 'BB_Low', '7_day_avg', '30_day_avg', 'Sentiment']
    
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression().fit(xtrain, ytrain)
    predictions = reg.predict(xtest)
    
    mae = mean_absolute_error(ytest, predictions)
    mse = mean_squared_error(ytest, predictions)
    r2 = r2_score(ytest, predictions)
    
    cross_val_scores = cross_val_score(reg, X, y, cv=5, scoring='r2')
    return reg, mae, mse, r2, np.mean(cross_val_scores), np.std(cross_val_scores)

def predict_next_day(reg, df):
    """Predict next day's price"""
    last_row = df.iloc[-1]
    next_day_features = np.array([[last_row['Date_Ordinal'] + 1, 
                                 last_row['Open'], 
                                 last_row['High'], 
                                 last_row['Low'], 
                                 last_row['Volume'], 
                                 last_row['RSI'], 
                                 last_row['BB_High'], 
                                 last_row['BB_Low'], 
                                 last_row['7_day_avg'], 
                                 last_row['30_day_avg'], 
                                 last_row['Sentiment']]])
    return reg.predict(next_day_features)[0]

def visualize_predictions(df, next_day_prediction):
    """Create interactive Plotly visualization"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Actual Prices'))
    future_date = df['Date'].iloc[-1] + pd.Timedelta(days=1)
    fig.add_trace(go.Scatter(x=[future_date], y=[next_day_prediction], 
                     mode='markers', name='Predicted Price'))
    fig.update_layout(title="Stock Price Prediction", xaxis_title="Date", 
                    yaxis_title="Price", template="plotly_white")
    st.plotly_chart(fig)

def linear_regression_prediction():
    """Linear Regression Prediction Interface"""
    st.title("Stock Price Prediction App - Linear Regression")
    
    with st.sidebar:
        st.header("Stock Selection")
        ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT):").upper()
        
        if ticker:
            start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
            end_date = st.date_input("End Date", value=datetime.today())
    
    if ticker and start_date < end_date:
        df = get_stock_data(ticker, start_date, end_date)
        if not df.empty:
            df['Sentiment'] = fetch_news_sentiment(ticker, NEWS_API_KEY)
            
            st.subheader("Stock Data")
            st.dataframe(df)
            
            reg, mae, mse, r2, mean_r2, std_r2 = train_and_evaluate_model(df)
            
            st.subheader("Model Performance")
            cols = st.columns(2)
            cols[0].metric("MAE", f"{mae:.2f}")
            cols[1].metric("R² Score", f"{r2:.2f}")
            
            prediction = predict_next_day(reg, df)
            st.subheader(f"Next Trading Day Prediction: ${prediction:.2f}")
            visualize_predictions(df, prediction)

# =========================
# LSTM Prediction Functions
# =========================

def lstm_prediction():
    """LSTM Prediction Interface"""
    st.title("Enhanced Stock Prediction - LSTM")
    
    with st.sidebar:
        st.header("Stock Selection")
        ticker = st.text_input("Enter Ticker Symbol:").upper()
        
        if ticker:
            start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
            end_date = st.date_input("End Date", value=datetime.today())
            future_date = st.date_input("Predict Until", value=datetime.today() + timedelta(days=30))
            start_button = st.button("Start Prediction")
    
    if ticker and start_date < end_date and start_button:
        data = get_stock_data(ticker, start_date, end_date)
        if not data.empty:
            data.set_index('Date', inplace=True)
            
            # Prepare data
            features = ['Close', 'Volume', 'MA50', 'MA200', 'RSI', 'MACD']
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data[features])
            
            # Build LSTM model
            time_step = 60
            X, y = [], []
            for i in range(len(scaled_data)-time_step-1):
                X.append(scaled_data[i:i+time_step])
                y.append(scaled_data[i+time_step, 0])
            X, y = np.array(X), np.array(y)
            
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            with st.spinner("Training Model..."):
                model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            
            # Generate predictions
            future_days = (future_date - end_date).days
            last_sequence = scaled_data[-time_step:]
            predictions = []
            
            for _ in range(future_days):
                pred = model.predict(last_sequence.reshape(1, time_step, len(features)))
                predictions.append(pred[0,0])
                last_sequence = np.roll(last_sequence, -1, axis=0)
                last_sequence[-1] = pred
                
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1,1)
            predictions = scaler.inverse_transform(
                np.hstack([predictions, np.zeros(len(predictions), len(features)-1)]))[:,0]
            
            # Create visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical'))
            future_dates = pd.date_range(start=end_date + timedelta(days=1), periods=future_days)
            fig.add_trace(go.Scatter(x=future_dates, y=predictions, name='Forecast'))
            fig.update_layout(title=f"{ticker} Price Forecast", xaxis_title="Date", 
                            yaxis_title="Price", template="plotly_dark")
            st.plotly_chart(fig)

# =========================
# Main Application
# =========================

def main():
    st.set_page_config(page_title="Stock Prediction App", layout="wide")
    st.title("AI Stock Market Predictor")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Short-Term Prediction (Linear Regression)"):
            st.session_state.page = "linear"
    with col2:
        if st.button("Long-Term Forecast (LSTM)"):
            st.session_state.page = "lstm"
    
    if "page" in st.session_state:
        if st.session_state.page == "linear":
            linear_regression_prediction()
        elif st.session_state.page == "lstm":
            lstm_prediction()

if __name__ == "__main__":
    main()