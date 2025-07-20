import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import nltk
import os

nltk.download('vader_lexicon')

# Streamlit page configuration
st.set_page_config(page_title="Enhanced Stock Prediction App", layout="wide")

# Initialize NewsAPI client (replace with your API key)
NEWS_API_KEY = "71e6d08d6731464e80b52967917470b0"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Predefined list of Indian stock tickers
INDIAN_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
    "SBIN.NS", "HDFC.NS", "ITC.NS", "LT.NS", "AXISBANK.NS"
]

# Function to fetch stock's first available date
def get_stock_first_date(ticker):
    stock_info = yf.Ticker(ticker).history(period="max")
    if not stock_info.empty:
        return stock_info.index.min().date()
    return None

# Function to fetch stock data, excluding weekends (Saturday and Sunday)
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Exclude weekends (Saturday and Sunday)
    data = data[(data.index.dayofweek != 5) & (data.index.dayofweek != 6)]

    # Add technical indicators
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    bb = ta.volatility.BollingerBands(data['Close'])
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()

    data.dropna(inplace=True)
    return data

# Function to create dataset with time steps
def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 0])  # Predict 'Close' price
    return np.array(X), np.array(y)

# Function to prepare data for LSTM
def prepare_data(data, time_step=120):
    features = ['Close', 'Volume', 'MA50', 'MA200', 'EMA12', 'EMA26', 'RSI', 'MACD', 'BB_High', 'BB_Low']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    return X_train, y_train, X_test, y_test, scaler

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(150, return_sequences=True, input_shape=input_shape),
        LSTM(100, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Function to fetch and analyze sentiment of news articles
def fetch_sentiment(ticker):
    company_name = ticker.split('.')[0]
    articles = newsapi.get_everything(q=company_name, language='en', sort_by='relevancy', page_size=20)

    if articles['totalResults'] == 0:
        return "No news articles available.", None

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    headlines = []

    for article in articles['articles']:
        headline = article['title']
        sentiment = analyzer.polarity_scores(headline)
        sentiment_scores.append(sentiment['compound'])
        headlines.append(headline)

    avg_sentiment = np.mean(sentiment_scores)
    return avg_sentiment, headlines

# Function to predict future prices using the trained model
def predict_future_prices(model, last_sequence, scaler, future_days):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_days):
        next_prediction = model.predict(current_sequence[np.newaxis, :, :])
        future_predictions.append(next_prediction[0, 0])
        new_sequence = np.roll(current_sequence, -1, axis=0)
        new_sequence[-1, 0] = next_prediction[0, 0]
        current_sequence = new_sequence

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_full = np.column_stack((future_predictions, np.zeros((len(future_predictions), 9))))
    future_predictions_inverse = scaler.inverse_transform(future_predictions_full)[:, 0]

    return future_predictions_inverse

# Function to plot predictions
def plot_predictions(data, y_test_actual, test_predict, ticker, scaler, future_predictions=None, future_date=None):
    all_close_prices = data['Close'].values
    all_dates = data.index

    test_predict_full = np.column_stack((test_predict, np.zeros((len(test_predict), 9))))
    test_predict_inverse = scaler.inverse_transform(test_predict_full)[:, 0]

    predicted_prices = np.full(len(all_close_prices), np.nan)
    predicted_prices[-len(test_predict_inverse):] = test_predict_inverse

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=all_dates, y=all_close_prices, 
        mode='lines', name='Actual Prices', 
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=all_dates[-len(test_predict_inverse):], y=predicted_prices[-len(test_predict_inverse):], 
        mode='lines', name='Predicted Prices (Test)', 
        line=dict(color='red')
    ))

    if future_predictions is not None and future_date is not None:
        future_dates = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), end=future_date)
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_predictions, 
            mode='lines', name='Future Predictions', 
            line=dict(color='green', dash='dot')
        ))

    fig.update_layout(
        title=f"{ticker} Stock Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        legend_title="Legend",
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        font=dict(color="white"),
        autosize=True
    )
    return fig

# Function to train or load the model; the model filename is now unique per ticker.
def train_or_load_model(X_train, y_train, epochs, batch_size, early_stopping, ticker):
    # Create a unique model file name for each ticker.
    model_path = f"trained_model_{ticker.replace('.', '_')}.h5"
  
    if os.path.exists(model_path):
        model = load_model(model_path)
        is_new_model = False
    else:
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        is_new_model = True
        with st.spinner("Training the model..."):
            model.fit(
                X_train, y_train,
                epochs=epochs, batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stopping]
            )
        model.save(model_path)

    return model, is_new_model

# Function to create broker button HTML
def broker_button(name, url):
    button_html = f"""
    <a href="{url}" target="_blank">
        <button style="
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;">
            {name}
        </button>
    </a>
    """
    return button_html

# Main function
def main():
    st.markdown(
        """
        <style>
        body {
            background-color: #1e1e1e;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Enhanced Stock Prediction App")
    st.markdown("**Predict stock prices using advanced LSTM models and analyze sentiment from news headlines.**")

    # Sidebar for user input
    with st.sidebar:
        st.header("Stock Selection")
        ticker = st.text_input("Enter Stock Ticker Symbol")
        stock_first_date = get_stock_first_date(ticker)

        if stock_first_date:
            start_date = st.date_input("Start Date", value=stock_first_date, min_value=stock_first_date)
        else:
            start_date = st.date_input("Start Date", value=datetime(2010, 1, 1))

        end_date = st.date_input("End Date", value=datetime.today())

        st.header("Model Settings")
        time_step = st.slider("Time Step (Days)", min_value=30, max_value=180, value=120, step=10)
        epochs = st.slider("Epochs", min_value=10, max_value=200, value=50, step=10)
        batch_size = st.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)

        # Input for future date prediction
        future_date = st.date_input("Predict Price for Future Date", value=datetime.today() + timedelta(days=1))

        start_button = st.button("Start Prediction")
        
        # Broker Options section
        st.header("Broker Options")
        brokers = {
            "Zerodha": "https://zerodha.com",
            "Upstox": "https://upstox.com",
            "Angel One": "https://www.angelone.in",
            "ICICI Direct": "https://www.icicidirect.com",
            "5paisa": "https://www.5paisa.com",
            "HDFC Securities": "https://www.hdfcsec.com"
        }
        for name, url in brokers.items():
            st.markdown(broker_button(name, url), unsafe_allow_html=True)

    # Proceed only if the start button is pressed and a ticker is provided
    if start_button and ticker:
        data = get_stock_data(ticker, start_date, end_date)

        if data.empty:
            st.warning("No data available for the selected ticker and date range.")
            return

        st.subheader(f"Data for {ticker}") 
        st.dataframe(data)

        # Prepare data for training and testing
        X_train, y_train, X_test, y_test, scaler = prepare_data(data, time_step)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train or load the model (using ticker to create a unique model file)
        model, is_new_model = train_or_load_model(X_train, y_train, epochs, batch_size, early_stopping, ticker)

        with st.spinner("Fetching sentiment data..."):
            avg_sentiment, headlines = fetch_sentiment(ticker)

        if headlines:
            st.subheader("News Sentiment Analysis")
            st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
            sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
            st.write(f"**Overall Sentiment:** {sentiment_label}")

            with st.expander("View Latest Headlines"):
                for headline in headlines:
                    st.write(f"- {headline}")

        if is_new_model:
            st.success("Model has been newly trained and saved!")
        else:
            st.info("Loaded pre-trained model.")

        test_predict = model.predict(X_test)
        train_predict = model.predict(X_train)

        train_predict_full = np.column_stack((train_predict, np.zeros((len(train_predict), 9))))
        train_predict_inverse = scaler.inverse_transform(train_predict_full)[:, 0]

        y_train_actual_full = np.column_stack((y_train, np.zeros((len(y_train), 9))))
        y_train_actual_inverse = scaler.inverse_transform(y_train_actual_full)[:, 0]

        test_predict_full = np.column_stack((test_predict, np.zeros((len(test_predict), 9))))
        test_predict_inverse = scaler.inverse_transform(test_predict_full)[:, 0]

        y_test_actual_full = np.column_stack((y_test, np.zeros((len(y_test), 9))))
        y_test_actual_inverse = scaler.inverse_transform(y_test_actual_full)[:, 0]

        rmse = np.sqrt(mean_squared_error(y_test_actual_inverse, test_predict_inverse))
        st.sidebar.write(f"**RMSE:** {rmse:.2f}")

        # Predict future prices if a valid future date is provided
        future_predictions = None
        if future_date:
            future_dates = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), end=future_date)
            future_days = len(future_dates)
            if future_days > 0:
                last_sequence = X_test[-1]
                future_predictions = predict_future_prices(model, last_sequence, scaler, future_days)
                st.subheader(f"Predicted Price for {future_date}")
                st.write(f"₹{future_predictions[-1]:.2f}")

        fig = plot_predictions(data, y_test, test_predict, ticker, scaler, future_predictions, future_date)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
