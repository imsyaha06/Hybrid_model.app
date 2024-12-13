# import streamlit as st
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# import joblib
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import MinMaxScaler

# # Load saved models and scaler
# lstm_model = load_model('lstm_model.h5')
# lin_model = joblib.load('linear_model.pkl')
# scaler = joblib.load('scaler.pkl')

# st.title("Hybrid Machine Learning Model")

# # Upload CSV
# uploaded_file = st.file_uploader("Upload your stock data CSV file", type="csv")

# if uploaded_file:
#     data = pd.read_csv(uploaded_file)
#     st.write("Uploaded Data:")
#     st.write(data.head())

#     # Preprocessing
#     data['Date'] = pd.to_datetime(data['Date'])
#     data.set_index('Date', inplace=True)
#     data = data[['Close']]
    
#     # Scaling
#     data['Close'] = scaler.transform(data[['Close']])
    
#     # Generate future predictions
#     seq_length = 60
    
#     def create_sequences(data, seq_length=60):
#         X = []
#         for i in range(len(data) - seq_length):
#             X.append(data[i:i+seq_length])
#         return np.array(X)
    
#     X = create_sequences(data['Close'].values, seq_length)
#     X_test_lstm = X[-1].reshape(1, seq_length, 1)
    
#     # LSTM future predictions
#     lstm_future_predictions = []
#     last_sequence = X[-1].reshape(1, seq_length, 1)
#     for _ in range(10):
#         lstm_pred = lstm_model.predict(last_sequence)[0, 0]
#         lstm_future_predictions.append(lstm_pred)
#         lstm_pred_reshaped = np.array([[lstm_pred]]).reshape(1, 1, 1)
#         last_sequence = np.append(last_sequence[:, 1:, :], lstm_pred_reshaped, axis=1)
#     lstm_future_predictions = scaler.inverse_transform(np.array(lstm_future_predictions).reshape(-1, 1))
    
#     # Linear Regression future predictions
#     recent_data = data['Close'].values[-3:]
#     lin_future_predictions = []
#     for _ in range(10):
#         lin_pred = lin_model.predict(recent_data.reshape(1, -1))[0]
#         lin_future_predictions.append(lin_pred)
#         recent_data = np.append(recent_data[1:], lin_pred)
#     lin_future_predictions = scaler.inverse_transform(np.array(lin_future_predictions).reshape(-1, 1))
    
#     # Hybrid predictions
#     hybrid_future_predictions = (0.7 * lstm_future_predictions) + (0.3 * lin_future_predictions)
    
#     # Future dates
#     future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=10)
#     predictions_df = pd.DataFrame({
#         'Date': future_dates,
#         'LSTM Predictions': lstm_future_predictions.flatten(),
#         'Linear Regression Predictions': lin_future_predictions.flatten(),
#         'Hybrid Model Predictions': hybrid_future_predictions.flatten()
#     })
    
#     # Display predictions
#     st.write("Future Predictions:")
#     st.write(predictions_df)
    
#     # Plot predictions
#     st.line_chart(predictions_df.set_index('Date'))














import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Load saved models and scaler
lstm_model = load_model('lstm_model.h5')
lin_model = joblib.load('linear_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("Stock Price Prediction App")

# Sidebar with description
st.sidebar.title("About This App")
st.sidebar.info(
    "This hybrid machine learning model is built using LSTM and Linear Regression, "
    "trained on a stock price prediction dataset to forecast future stock prices. "
    "The LSTM model captures sequential patterns in the data, while the Linear Regression "
    "model adds complementary predictive capabilities."
)

# Upload CSV
uploaded_file = st.file_uploader("Upload your stock data CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data.head())

    # Preprocessing
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data[['Close']]

    # Scaling
    data['Close'] = scaler.transform(data[['Close']])

    # Generate future predictions
    seq_length = 60

    def create_sequences(data, seq_length=60):
        X = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
        return np.array(X)

    X = create_sequences(data['Close'].values, seq_length)
    X_test_lstm = X[-1].reshape(1, seq_length, 1)

    # LSTM future predictions
    lstm_future_predictions = []
    last_sequence = X[-1].reshape(1, seq_length, 1)
    for _ in range(10):
        lstm_pred = lstm_model.predict(last_sequence)[0, 0]
        lstm_future_predictions.append(lstm_pred)
        lstm_pred_reshaped = np.array([[lstm_pred]]).reshape(1, 1, 1)
        last_sequence = np.append(last_sequence[:, 1:, :], lstm_pred_reshaped, axis=1)
    lstm_future_predictions = scaler.inverse_transform(np.array(lstm_future_predictions).reshape(-1, 1))

    # Linear Regression future predictions
    recent_data = data['Close'].values[-3:]
    lin_future_predictions = []
    for _ in range(10):
        lin_pred = lin_model.predict(recent_data.reshape(1, -1))[0]
        lin_future_predictions.append(lin_pred)
        recent_data = np.append(recent_data[1:], lin_pred)
    lin_future_predictions = scaler.inverse_transform(np.array(lin_future_predictions).reshape(-1, 1))

    # Hybrid predictions
    hybrid_future_predictions = (0.7 * lstm_future_predictions) + (0.3 * lin_future_predictions)

    # Future dates
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=10)
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'LSTM Predictions': lstm_future_predictions.flatten(),
        'Linear Regression Predictions': lin_future_predictions.flatten(),
        'Hybrid Model Predictions': hybrid_future_predictions.flatten()
    })

    # Display predictions
    st.write("Future Predictions:")
    st.write(predictions_df)

    # Plot predictions
    st.line_chart(predictions_df.set_index('Date'))
