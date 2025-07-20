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
from tiingo import TiingoClient
import streamlit as st
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from newsapi import NewsApiClient

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Initialize Tiingo client
config = {
    'api_key': '6d4ca467b47470bf9c34e1fb0d06bfa4140656f2',
    'session': True
}
tiingo_client = TiingoClient(config)

# Initialize NewsAPI client
NEWS_API_KEY = "71e6d08d6731464e80b52967917470b0"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# LSTM Default Parameters
TIME_STEP = 120
EPOCHS = 100
BATCH_SIZE = 32
# =========================
# Helper Functions
# =========================

def get_stock_first_date(ticker):
    try:
        metadata = tiingo_client.get_ticker_metadata(ticker)
        start_date = metadata['startDate']
        return pd.to_datetime(start_date).tz_localize(None).date()
    except Exception as e:
        st.error(f"Error fetching metadata for {ticker}: {e}")
        return None

def get_stock_data(ticker, start_date, end_date):
    data = tiingo_client.get_dataframe(ticker, startDate=start_date, endDate=end_date, frequency='daily')
    
    if data.empty:
        st.warning("No data available for the selected ticker and date range.")
        return pd.DataFrame()
    
    data = data.reset_index()
    data.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    data = data[(data['Date'].dt.dayofweek != 5) & (data['Date'].dt.dayofweek != 6)]
    
    # Technical indicators
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    bb = ta.volatility.BollingerBands(data['Close'])
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    data['7_day_avg'] = data['Close'].rolling(window=7).mean()
    data['30_day_avg'] = data['Close'].rolling(window=30).mean()
    
    data.dropna(inplace=True)
    return data

def fetch_news_sentiment(ticker):
    try:
        company_name = ticker.split('.')[0]
        articles = newsapi.get_everything(q=company_name, language='en', sort_by='relevancy', page_size=20)
        
        if articles['totalResults'] == 0:
            return 0, ["No news articles available"]
            
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
        
    except Exception as e:
        st.error(f"Error fetching news sentiment: {e}")
        return 0, ["Error fetching news"]

# =========================
# Linear Regression Functions
# =========================

def train_and_evaluate_linear_model(df, sentiment_score):
    df['Date_Ordinal'] = df['Date'].map(lambda x: x.toordinal())
    df['Sentiment'] = sentiment_score
    
    X = df[['Date_Ordinal', 'Open', 'High', 'Low', 'Volume', 
            'RSI', 'BB_High', 'BB_Low', '7_day_avg', '30_day_avg', 'Sentiment']]
    y = df['Close']
    
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression().fit(xtrain, ytrain)
    predictions = reg.predict(xtest)
    
    # Standard metrics
    mae = mean_absolute_error(ytest, predictions)
    mse = mean_squared_error(ytest, predictions)
    r2 = r2_score(ytest, predictions)
    
    # Accuracy calculations
    percentage_diff = np.abs((predictions - ytest) / ytest)
    accuracy_percentage = (1 - np.mean(percentage_diff)) * 100
    
    # Directional accuracy
    actual_direction = np.sign(np.diff(ytest))
    predicted_direction = np.sign(np.diff(predictions))
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    cross_val_scores = cross_val_score(reg, X, y, cv=5, scoring='r2')
    mean_r2 = np.mean(cross_val_scores)
    std_r2 = np.std(cross_val_scores)
    
    return reg, mae, mse, r2, accuracy_percentage, directional_accuracy, mean_r2, std_r2

def predict_next_day(reg, df, sentiment_score):
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
                                 sentiment_score]])
    next_day_prediction = reg.predict(next_day_features)
    return next_day_prediction[0]

def visualize_linear_predictions(df, next_day_prediction):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Actual Prices',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[-1] + pd.Timedelta(days=1)],
        y=[next_day_prediction],
        mode='markers',
        name='Predicted Price',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title="Stock Price Prediction",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend_title="Legend",
        hovermode="x unified",
        template="plotly_white"
    )
    
    st.plotly_chart(fig)

# =========================
# LSTM Functions
# =========================

def create_dataset(data_array, time_step):
    X, y = [], []
    for i in range(len(data_array) - time_step - 1):
        X.append(data_array[i:(i+time_step), :])
        y.append(data_array[i+time_step, 0])
    return np.array(X), np.array(y)

def prepare_lstm_data(data, time_step=120):
    features = ['Close', 'Volume', 'MA50', 'MA200', 'EMA12', 'EMA26', 'RSI', 'MACD', 'BB_High', 'BB_Low']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    return X_train, y_train, X_test, y_test, scaler

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(150, return_sequences=True, input_shape=input_shape),
        LSTM(100, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_or_load_lstm_model(X_train, y_train, epochs, batch_size, early_stopping, ticker):
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

def visualize_lstm_predictions(data, y_test_actual, test_predict, ticker, scaler, future_predictions=None, future_date=None):
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
    
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Main App
# =========================

def show_broker_options():
    st.header("Broker Options")
    brokers = {
        "Zerodha": "https://zerodha.com",
        "Upstox": "https://upstox.com",
        "Angel One": "https://www.angelone.in",
        "ICICI Direct": "https://www.icicidirect.com",
        "5paisa": "https://www.5paisa.com",
        "HDFC Securities": "https://www.hdfcsec.com"
    }
    
    cols = st.columns(3)
    for i, (broker, url) in enumerate(brokers.items()):
        button_html = f'''
        <a href="{url}" target="_blank">
            <div style="background-color:#0066CC; color:white; padding:10px 24px; border-radius:5px; text-align:center; margin:5px 0;">
                {broker}
            </div>
        </a>
        '''
        cols[i % 3].markdown(button_html, unsafe_allow_html=True)

def show_sentiment_analysis(avg_sentiment, headlines):
    st.subheader("News Sentiment Analysis")
    st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f}")
    sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
    st.write(f"**Overall Sentiment:** {sentiment_label}")
    with st.expander("View Latest Headlines"):
        for headline in headlines:
            st.write(f"- {headline}")

def main():
    st.set_page_config(page_title="MarketMinds", layout="wide")
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Stock Prediction Settings")
        
        # Model selection
        model_type = st.radio(
            "Select Prediction Model",
            ("Linear Regression (1-day)", "LSTM (Long-term)"),
            index=0
        )
        
        # Common inputs
        ticker = st.text_input("Stock Ticker Symbol", "AAPL").upper()
        
        stock_first_date = get_stock_first_date(ticker)
        if stock_first_date:
            start_date = st.date_input("Start Date", value=stock_first_date, min_value=stock_first_date)
        else:
            start_date = st.date_input("Start Date", value=datetime(2010, 1, 1))
        
        end_date = st.date_input("End Date", value=datetime.today())
        
        # Only show future date for LSTM
        if model_type == "LSTM (Long-term)":
            future_date = st.date_input("Predict Price for Future Date", value=datetime.today() + timedelta(days=7))
        
        predict_button = st.button("Run Prediction")
    
    # Main content area
    st.title("MarketMinds")
    
    if predict_button and ticker:
        if end_date < start_date:
            st.error("End date must be after the start date.")
            return
        
        # Get stock data
        with st.spinner("Fetching stock data..."):
            data = get_stock_data(ticker, start_date, end_date)
            if data.empty:
                st.error("No data available for the selected date range.")
                return
        
        # Get sentiment data
        with st.spinner("Analyzing news sentiment..."):
            avg_sentiment, headlines = fetch_news_sentiment(ticker)
        
        # Show sentiment analysis for both models
        show_sentiment_analysis(avg_sentiment, headlines)
        
        with st.expander("View Stock Data"):
            st.dataframe(data)
        
        if model_type == "Linear Regression (1-day)":
            st.header("Linear Regression Prediction")
            
            reg, mae, mse, r2, accuracy_percentage, directional_accuracy, mean_r2, std_r2 = train_and_evaluate_linear_model(data, avg_sentiment)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Absolute Error", f"{mae:.4f}")
                st.metric("Mean Squared Error", f"{mse:.4f}")
                st.metric("R² Score", f"{r2:.4f}")
            with col2:
                st.metric("Accuracy Percentage", f"{accuracy_percentage:.2f}%")
        
            next_day_prediction = predict_next_day(reg, data, avg_sentiment)
            st.subheader("Next Day Prediction")
            st.success(f"Predicted price for next trading day: ${next_day_prediction:.2f}")
            
            visualize_linear_predictions(data, next_day_prediction)
            
        elif model_type == "LSTM (Long-term)":
            st.header("LSTM Prediction")
            
            data.set_index('Date', inplace=True)
            X_train, y_train, X_test, y_test, scaler = prepare_lstm_data(data, TIME_STEP)
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model, is_new_model = train_or_load_lstm_model(X_train, y_train, EPOCHS, BATCH_SIZE, early_stopping, ticker)
            
            # Make predictions
            test_predict = model.predict(X_test)
            
            # Inverse transform predictions
            test_predict_full = np.column_stack((test_predict, np.zeros((len(test_predict), 9))))
            test_predict_inverse = scaler.inverse_transform(test_predict_full)[:, 0]
            y_test_actual_full = np.column_stack((y_test, np.zeros((len(y_test), 9))))
            y_test_actual_inverse = scaler.inverse_transform(y_test_actual_full)[:, 0]
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test_actual_inverse, test_predict_inverse))
            mae = mean_absolute_error(y_test_actual_inverse, test_predict_inverse)
            r2 = r2_score(y_test_actual_inverse, test_predict_inverse)
            
            # Accuracy calculations
            percentage_diff = np.abs((test_predict_inverse - y_test_actual_inverse) / y_test_actual_inverse)
            accuracy_percentage = (1 - np.mean(percentage_diff)) * 100
            
            # Directional accuracy
            actual_direction = np.sign(np.diff(y_test_actual_inverse))
            predicted_direction = np.sign(np.diff(test_predict_inverse))
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE", f"{rmse:.2f}")
                st.metric("MAE", f"{mae:.2f}")
                st.metric("R² Score", f"{r2:.4f}")
            with col2:
                st.metric("Accuracy Percentage", f"{accuracy_percentage:.2f}%")
                
            
            # Future predictions
            future_predictions = None
            if model_type == "LSTM (Long-term)":
                last_date = data.index[-1]
                future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), end=future_date)
                future_days = len(future_dates)
                
                if future_days > 0:
                    last_sequence = X_test[-1]
                    future_predictions = predict_future_prices(model, last_sequence, scaler, future_days)
                    st.subheader(f"Predicted Price for {future_date}")
                    st.success(f"${future_predictions[-1]:.2f}")
            
            visualize_lstm_predictions(data, y_test, test_predict, ticker, scaler, future_predictions, future_date)
    
    # Show broker options on main screen
    show_broker_options()

if __name__ == "__main__":
    main()
 