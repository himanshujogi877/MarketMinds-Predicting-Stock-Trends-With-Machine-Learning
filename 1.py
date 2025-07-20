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
    'api_key': '7ecfae35ea8b69201901bb09f896a73c5aacc8c2',  # Replace with your Tiingo API key
    'session': True
}
tiingo_client = TiingoClient(config)

# Initialize NewsAPI client
NEWS_API_KEY = "71e6d08d6731464e80b52967917470b0"  # Replace with your NewsAPI key
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# =========================
# Helper Functions
# =========================

def get_stock_first_date(ticker):
    """
    Fetch the earliest available date for the stock using Tiingo metadata.
    """
    try:
        # Fetch metadata for the stock
        metadata = tiingo_client.get_ticker_metadata(ticker)
        start_date = metadata['startDate']  # Earliest available date
        return pd.to_datetime(start_date).tz_localize(None).date()  # Remove timezone
    except Exception as e:
        st.error(f"Error fetching metadata for {ticker}: {e}")
        return None

def get_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Tiingo and preprocess it.
    """
    # Fetch data from Tiingo
    data = tiingo_client.get_dataframe(ticker, startDate=start_date, endDate=end_date, frequency='daily')
    
    # Check if data is empty
    if data.empty:
        st.warning("No data available for the selected ticker and date range.")
        return pd.DataFrame()
    
    # Reset index to make 'Date' a column
    data = data.reset_index()
    
    # Rename columns to match expected format
    data.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    
    # Ensure 'Date' is in datetime format and timezone-naive
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)  # Remove timezone
    
    # Filter out weekends (if needed)
    data = data[(data['Date'].dt.dayofweek != 5) & (data['Date'].dt.dayofweek != 6)]
    
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
    
    # Add 7-day and 30-day rolling averages
    data['7_day_avg'] = data['Close'].rolling(window=7).mean()
    data['30_day_avg'] = data['Close'].rolling(window=30).mean()
    
    # Drop rows with NaN values (from technical indicators)
    data.dropna(inplace=True)
    
    return data

def fetch_news_sentiment(ticker, api_key):
    """
    Fetch news sentiment using NewsAPI.
    """
    url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}'
    response = requests.get(url)
    news_data = response.json()
    
    if news_data['status'] == 'ok':
        headlines = [article['title'] for article in news_data['articles']]
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
        return np.mean(sentiment_scores) if sentiment_scores else 0
    else:
        return 0  # Return neutral sentiment if no data is found

# =========================
# Linear Regression Functions
# =========================

def train_and_evaluate_model(df):
    """
    Train and evaluate the Linear Regression model.
    """
    df['Date_Ordinal'] = df['Date'].map(lambda x: x.toordinal())
    X = df[['Date_Ordinal', 'Open', 'High', 'Low', 'Volume', 
            'RSI', 'BB_High', 'BB_Low', '7_day_avg', '30_day_avg', 'Sentiment']]
    y = df['Close']
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression().fit(xtrain, ytrain)
    predictions = reg.predict(xtest)
    
    mae = mean_absolute_error(ytest, predictions)
    mse = mean_squared_error(ytest, predictions)
    r2 = r2_score(ytest, predictions)
    
    cross_val_scores = cross_val_score(reg, X, y, cv=5, scoring='r2')
    mean_r2 = np.mean(cross_val_scores)
    std_r2 = np.std(cross_val_scores)
    
    return reg, mae, mse, r2, mean_r2, std_r2

def predict_next_day(reg, df):
    """
    Predict the next day's stock price.
    """
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
    next_day_prediction = reg.predict(next_day_features)
    return next_day_prediction[0]

def visualize_predictions(df, next_day_prediction):
    """
    Visualize actual and predicted prices using Plotly.
    """
    fig = go.Figure()
    
    # Plot actual prices
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Actual Prices',
        line=dict(color='blue')
    ))
    
    # Plot predicted price for the next day
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[-1] + pd.Timedelta(days=1)],
        y=[next_day_prediction],
        mode='markers',
        name='Predicted Price',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title="Stock Price Prediction with Enhanced Features",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        legend_title="Legend",
        hovermode="x unified",
        template="plotly_white"
    )
    
    st.plotly_chart(fig)

def linear_regression_prediction():
    st.title("Stock Price Prediction App - Linear Regression")
    ticker = st.text_input("Enter stock ticker (e.g., 'AAPL', 'GOOG', 'HDFCBANK.NS'):").upper()
    
    if ticker:
        # Fetch the earliest available date for the stock
        stock_first_date = get_stock_first_date(ticker)
        
        # Set default start date to the earliest available date
        if stock_first_date:
            start_date = st.date_input("Select Start Date", value=stock_first_date, min_value=stock_first_date)
        else:
            start_date = st.date_input("Select Start Date", value=datetime(2010, 1, 1))  # Fallback to a default date
        
        end_date = st.date_input("Select End Date", value=datetime.today())
        
        if end_date < start_date:
            st.error("End date must be after the start date.")
            return
        
        # Fetch stock data
        df = get_stock_data(ticker, start_date, end_date)
        if df.empty:
            st.error("No data available for the selected date range. Please adjust your range.")
            return
        
        # Fetch sentiment score
        sentiment_score = fetch_news_sentiment(ticker, NEWS_API_KEY)
        df['Sentiment'] = sentiment_score  # Add sentiment score to the DataFrame
        
        st.subheader("Stock Data")
        st.write(df)
        
        # Train the model and display evaluation metrics
        reg, mae, mse, r2, mean_r2, std_r2 = train_and_evaluate_model(df)
        st.subheader("Model Evaluation Metrics")
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"R² Score: {r2:.4f}")
        st.write(f"Cross-validation R² mean: {mean_r2:.4f}")
        st.write(f"Cross-validation R² std deviation: {std_r2:.4f}")
        
        # Predict the next trading day's price
        next_day_prediction = predict_next_day(reg, df)
        st.subheader("Predicted Stock Price for Next Trading Day")
        st.write(f"Predicted stock price for the next trading day: ₹{next_day_prediction:.2f}")
        
        # Visualize the prediction results using Plotly
        visualize_predictions(df, next_day_prediction)
        
        # Broker Options Section using clickable links styled as buttons
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

# =========================
# LSTM Prediction Functions
# =========================

def lstm_prediction():
    st.title("Enhanced Stock Prediction App - LSTM")
    
    # Default values for hidden parameters
    TIME_STEP = 120  # Default time step
    BATCH_SIZE = 32  # Default batch size
    
    # Sidebar for user input
    with st.sidebar:
        st.header("Stock Selection")
        ticker = st.text_input("Enter Stock Ticker Symbol")
        
        # Fetch the earliest available date for the stock
        stock_first_date = get_stock_first_date(ticker) if ticker else None
        
        # Set default start date to the earliest available date
        if stock_first_date:
            start_date = st.date_input("Start Date", value=stock_first_date, min_value=stock_first_date)
        else:
            start_date = st.date_input("Start Date", value=datetime(2010, 1, 1))  # Fallback to a default date
        
        end_date = st.date_input("End Date", value=datetime.today())
        
        st.header("Model Settings")
        epochs = st.slider("Epochs", min_value=10, max_value=200, value=50, step=10)
        future_date = st.date_input("Predict Price for Future Date", value=datetime.today() + timedelta(days=1))
        start_button = st.button("Start Prediction")
        
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
    
    if start_button and ticker:
        data = get_stock_data(ticker, start_date, end_date)
        if data.empty:
            st.warning("No data available for the selected ticker and date range.")
            return
        
        # Set 'Date' as the index
        data.set_index('Date', inplace=True)
        
        st.subheader(f"Data for {ticker}")
        st.dataframe(data)
        
        # Prepare data for LSTM
        def create_dataset(data_array, time_step):
            X, y = [], []
            for i in range(len(data_array) - time_step - 1):
                X.append(data_array[i:(i+time_step), :])
                y.append(data_array[i+time_step, 0])
            return np.array(X), np.array(y)
        
        def prepare_data(data, time_step=TIME_STEP):
            features = ['Close', 'Volume', 'MA50', 'MA200', 'EMA12', 'EMA26', 'RSI', 'MACD', 'BB_High', 'BB_Low']
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data[features])
            train_size = int(len(scaled_data) * 0.8)
            train_data = scaled_data[:train_size]
            test_data = scaled_data[train_size:]
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, y_test = create_dataset(test_data, time_step)
            return X_train, y_train, X_test, y_test, scaler
        
        X_train, y_train, X_test, y_test, scaler = prepare_data(data)
        
        # Build and train LSTM model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        def build_lstm_model(input_shape):
            model = Sequential([
                LSTM(150, return_sequences=True, input_shape=input_shape),
                LSTM(100, return_sequences=True),
                LSTM(50, return_sequences=False),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            return model
        
        def train_or_load_model(X_train, y_train, epochs, batch_size, early_stopping, ticker):
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
        
        model, is_new_model = train_or_load_model(X_train, y_train, epochs, BATCH_SIZE, early_stopping, ticker)
        
        # Fetch sentiment data
        with st.spinner("Fetching sentiment data..."):
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
            
            avg_sentiment, headlines = fetch_sentiment(ticker)
        
        if headlines:
            st.subheader("News Sentiment Analysis")
            st.write(f"*Average Sentiment Score:* {avg_sentiment:.2f}")
            sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
            st.write(f"*Overall Sentiment:* {sentiment_label}")
            with st.expander("View Latest Headlines"):
                for headline in headlines:
                    st.write(f"- {headline}")
        
        if is_new_model:
            st.success("Model has been newly trained and saved!")
        else:
            st.info("Loaded pre-trained model.")
        
        # Make predictions
        test_predict = model.predict(X_test)
        train_predict = model.predict(X_train)
        
        # Inverse transform predictions
        train_predict_full = np.column_stack((train_predict, np.zeros((len(train_predict), 9))))
        train_predict_inverse = scaler.inverse_transform(train_predict_full)[:, 0]
        y_train_actual_full = np.column_stack((y_train, np.zeros((len(y_train), 9))))
        y_train_actual_inverse = scaler.inverse_transform(y_train_actual_full)[:, 0]
        test_predict_full = np.column_stack((test_predict, np.zeros((len(test_predict), 9))))
        test_predict_inverse = scaler.inverse_transform(test_predict_full)[:, 0]
        y_test_actual_full = np.column_stack((y_test, np.zeros((len(y_test), 9))))
        y_test_actual_inverse = scaler.inverse_transform(y_test_actual_full)[:, 0]
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test_actual_inverse, test_predict_inverse))
        st.sidebar.write(f"*RMSE:* {rmse:.2f}")
        
        # Predict future prices
        future_predictions = None
        if future_date:
            # Ensure the last date in the data is a datetime object
            last_date = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['Date'].iloc[-1])
            
            # Calculate future dates
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), end=future_date)
            future_days = len(future_dates)
            
            if future_days > 0:
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
                
                last_sequence = X_test[-1]
                future_predictions = predict_future_prices(model, last_sequence, scaler, future_days)
                st.subheader(f"Predicted Price for {future_date}")
                st.write(f"₹{future_predictions[-1]:.2f}")
        
        # Plot predictions
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
        
        fig = plot_predictions(data, y_test, test_predict, ticker, scaler, future_predictions, future_date)
        st.plotly_chart(fig, use_container_width=True)

# =========================
# Main Function
# =========================

def main():
    st.set_page_config(page_title="Stock Price Prediction App", layout="wide")
    st.title("Stock Price Prediction App")
    st.write("Choose the type of prediction you want to perform:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("One-Day Prediction (Linear Regression)"):
            st.session_state.prediction_type = "linear"
    with col2:
        if st.button("Long-Term Prediction (LSTM)"):
            st.session_state.prediction_type = "lstm"
    if "prediction_type" in st.session_state:
        if st.session_state.prediction_type == "linear":
            linear_regression_prediction()
        elif st.session_state.prediction_type == "lstm":
            lstm_prediction()

if __name__ == "__main__":
    main()