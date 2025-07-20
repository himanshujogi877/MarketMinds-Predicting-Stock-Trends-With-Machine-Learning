import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Download TCS stock price data from Yahoo Finance for the past 10 years
ticker = 'TCS.NS'  # TCS stock ticker on NSE
data = yf.download(ticker, start='2013-01-01', end='2024-10-28')

# Extract relevant columns
df = pd.DataFrame(data['Close'])  # 'Close' column has the closing stock prices
df = df.reset_index()  # Reset index to move the Date column into the dataframe

# Handle any missing values in 'Close' price
df['Close'] = df['Close'].fillna(df['Close'].median())

# Create additional features: 7-day and 30-day moving averages
df['7_day_avg'] = df['Close'].rolling(window=7).mean()
df['30_day_avg'] = df['Close'].rolling(window=30).mean()

# Drop rows with NaN values generated from the rolling averages
df = df.dropna()

# Graph 1: Plot TCS stock data for the past 10 years with moving averages
plt.figure(figsize=(10,6))
plt.plot(df['Date'], df['Close'], label='TCS Closing Price')
plt.plot(df['Date'], df['7_day_avg'], label='7-Day Moving Average', linestyle='--')
plt.plot(df['Date'], df['30_day_avg'], label='30-Day Moving Average', linestyle='--')
plt.title('TCS Stock Price for the Past 10 Years')
plt.xlabel('Date')
plt.ylabel('Stock Price (INR)')
plt.legend()
plt.grid(True)
plt.show()

### Prepare the data for training ###
# Use 'Date', '7_day_avg', and '30_day_avg' as features (independent variables)
df['Date'] = pd.to_datetime(df['Date'])
X = df[['Date', '7_day_avg', '30_day_avg']]

# Convert date to ordinal (integer representation) for model
X['Date'] = X['Date'].map(datetime.toordinal)

# 'Close' as the dependent variable
y = df['Close']

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
reg = LinearRegression().fit(xtrain, ytrain)

# Evaluate the model's performance on the test set
accuracy = reg.score(xtest, ytest)
print(f"Model Accuracy: {accuracy}")

# Save the trained model
with open('tcs_stock_prediction.pickle', 'wb') as f:
    pickle.dump(reg, f)

# Graph 2: Plot this week's stock data and predict the next trading day
# Get the last 7 trading days (excluding weekends/holidays)
last_week_data = df[-7:]  # Last 7 trading days

# Prepare the last week's data for prediction
last_week_features = last_week_data[['Date', '7_day_avg', '30_day_avg']]
last_week_features['Date'] = last_week_features['Date'].map(datetime.toordinal)

# Predict the stock price for the next trading day (market open)
next_trading_day = last_week_data['Date'].max() + timedelta(days=1)
while next_trading_day.weekday() >= 5:  # Skip weekends (Saturday=5, Sunday=6)
    next_trading_day += timedelta(days=1)

next_day_ordinal = np.array([[next_trading_day.toordinal(), last_week_data['7_day_avg'].values[-1], last_week_data['30_day_avg'].values[-1]]])
predicted_price = reg.predict(next_day_ordinal)

print(f"Predicted stock price for {next_trading_day.date()}: {predicted_price[0]}")

# Plot this week's stock prices with the predicted price for the next trading day
plt.figure(figsize=(10,6))

# Plot the actual prices of this week
plt.plot(last_week_data['Date'], last_week_data['Close'], label='This Week\'s Stock Prices', marker='o', color='green')

# Plot the predicted price for the next trading day
plt.scatter(next_trading_day, predicted_price, color='red', label=f'Predicted Price for {next_trading_day.date()}', zorder=5)

# Improve x-axis labels (use actual dates)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.gcf().autofmt_xdate()  # Auto-format the date labels to prevent overlap

plt.title('TCS Stock Prices (This Week) + Next Trading Day Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price (INR)')
plt.legend()
plt.grid(True)
plt.show()
