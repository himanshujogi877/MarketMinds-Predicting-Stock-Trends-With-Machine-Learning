import streamlit as st
import yfinance as yf
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

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Function to download stock data and preprocess it by adding technical indicators
def get_data(ticker):
    data = yf.download(ticker, period="max")
    if data.empty:
        return pd.DataFrame()
    
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df = df.reset_index()
    # Ensure the Date column is datetime type
    df['Date'] = pd.to_datetime(df['Date'])
    df.fillna(method='ffill', inplace=True)
    
    # Add technical indicators
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    bb_indicator = ta.volatility.BollingerBands(close=df['Close'], window=20)
    df['BB_High'] = bb_indicator.bollinger_hband()
    df['BB_Low'] = bb_indicator.bollinger_lband()
    df['7_day_avg'] = df['Close'].rolling(window=7).mean()
    df['30_day_avg'] = df['Close'].rolling(window=30).mean()
    df.dropna(inplace=True)
    return df

# Function to fetch news sentiment using NewsAPI
def fetch_news_sentiment(ticker, api_key):
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

# Function to train and evaluate the linear regression model
def train_and_evaluate_model(df):
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

# Function to predict the next day's stock price
def predict_next_day(reg, df):
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

# Function to visualize the prediction results using Plotly
def visualize_predictions(df, next_day_prediction):
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

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Stock Price Prediction", layout="wide")
    st.title("Stock Price Prediction App")
    
    # Input for stock ticker
    ticker = st.text_input("Enter stock ticker (e.g., 'AAPL', 'GOOG', 'HDFCBANK.NS'):").upper()
    
    if ticker:
        df = get_data(ticker)
        if df.empty:
            st.error("No data found for the entered stock ticker. Please check the ticker symbol.")
            return
        
        # Determine the earliest and latest dates from the data
        earliest_date = df['Date'].min().date()
        latest_date = df['Date'].max().date()
        
        # Persist date inputs using session_state so they don't reset
        if 'start_date' not in st.session_state:
            st.session_state.start_date = earliest_date
        if 'end_date' not in st.session_state:
            st.session_state.end_date = latest_date
        
        start_date = st.date_input("Select Start Date", value=st.session_state.start_date)
        end_date = st.date_input("Select End Date", value=st.session_state.end_date)
        
        # Save selections back to session_state
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        
        if end_date < start_date:
            st.error("End date must be after the start date.")
            return
        
        # Filter the data based on the selected date range
        filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
        if filtered_df.empty:
            st.error("No data available for the selected date range. Please adjust your range.")
            return
        
        # Fetch sentiment score (replace API key with your own if needed)
        api_key = '71e6d08d6731464e80b52967917470b0'
        sentiment_score = fetch_news_sentiment(ticker, api_key)
        filtered_df['Sentiment'] = sentiment_score
        
        st.subheader("Stock Data")
        st.write(filtered_df)
        
        # Train the model and display evaluation metrics
        reg, mae, mse, r2, mean_r2, std_r2 = train_and_evaluate_model(filtered_df)
        st.subheader("Model Evaluation Metrics")
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"R² Score: {r2:.4f}")
        st.write(f"Cross-validation R² mean: {mean_r2:.4f}")
        st.write(f"Cross-validation R² std deviation: {std_r2:.4f}")
        
        # Predict the next trading day's price
        next_day_prediction = predict_next_day(reg, filtered_df)
        st.subheader("Predicted Stock Price for Next Trading Day")
        st.write(f"Predicted stock price for the next trading day: ₹{next_day_prediction:.2f}")
        
        # Visualize the prediction results using Plotly
        visualize_predictions(filtered_df, next_day_prediction)
        
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
                <div class="custom-button" style="background-color:#0066CC; color:white; padding:10px 24px; border-radius:5px; text-align:center; margin:5px 0;">
                    {broker}
                </div>
            </a>
            '''
            cols[i % 3].markdown(button_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
