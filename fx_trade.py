import os
from dotenv import load_dotenv
import paypalrestsdk
import logging
import smtplib
from email.mime.text import MIMEText
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load environment variables
load_dotenv()

# PayPal Configuration
paypalrestsdk.configure({
    "mode": os.getenv("PAYPAL_MODE", "sandbox"),
    "client_id": os.getenv("PAYPAL_CLIENT_ID"),
    "client_secret": os.getenv("PAYPAL_CLIENT_SECRET"),
})

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Parameters
profit_threshold = 1000
profit_to_transfer = 200
recipient_email = "recipient_email@example.com"

# Function: Predict trade signals using machine learning
def predict_trade_signals(data):
    # Placeholder ML model (RandomForest)
    model = RandomForestClassifier()
    
    # Simulate training data
    X_train = np.random.rand(100, 4)  # Random features
    y_train = np.random.choice([0, 1], size=100)  # Buy/Sell labels
    model.fit(X_train, y_train)

    # Predict on new data
    X_new = np.array(data).reshape(1, -1)  # Ensure input is 2D
    prediction = model.predict(X_new)[0]  # 0 = Sell, 1 = Buy
    return "Buy" if prediction == 1 else "Sell"

# Function: Get real-time profit
def get_real_time_profit(trading_data):
    total_revenue = sum([trade['profit'] for trade in trading_data])
    total_costs = sum([trade['cost'] for trade in trading_data])
    return total_revenue - total_costs

# Function: Transfer funds
def transfer_funds(amount):
    try:
        payout = paypalrestsdk.Payout({
            "sender_batch_header": {
                "sender_batch_id": "batch_" + str(int(amount)),
                "email_subject": "Profit Transfer Notification"
            },
            "items": [
                {
                    "recipient_type": "EMAIL",
                    "amount": {"value": f"{amount}", "currency": "USD"},
                    "receiver": recipient_email,
                    "note": "Automated profit transfer"
                }
            ]
        })

        if payout.create(sync_mode=True):
            logging.info(f"Payout created successfully. Batch ID: {payout.batch_header.payout_batch_id}")
            return True
        else:
            logging.error(f"Error creating payout: {payout.error}")
            return False
    except Exception as e:
        logging.error(f"Exception during payout: {e}")
        return False

# Function: Send email notifications
def send_email_notification(amount):
    sender_email = "your_email@example.com"
    sender_password = "your_email_password"
    smtp_server = "smtp.gmail.com"
    port = 587

    try:
        msg = MIMEText(f"A profit transfer of ${amount} has been sent to {recipient_email}.")
        msg["Subject"] = "Profit Transfer Notification"
        msg["From"] = sender_email
        msg["To"] = recipient_email

        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        logging.info("Notification email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email notification: {e}")

# Function: Monitor profit and automate transfer
def monitor_profit_and_transfer(trading_data):
    total_profit = get_real_time_profit(trading_data)

    if total_profit >= profit_threshold:
        logging.info(f"Profit threshold reached: ${total_profit}")
        if transfer_funds(profit_to_transfer):
            send_email_notification(profit_to_transfer)
        else:
            logging.error("Transfer failed.")
    else:
        logging.info(f"Current profit (${total_profit}) is below the threshold.")

# Example Usage
if __name__ == "__main__":
    # Simulated trading data
    trading_data = [
        {"trade_id": 1, "profit": 500, "cost": 50},
        {"trade_id": 2, "profit": 700, "cost": 100}
    ]

    # Example data for ML prediction
    new_trade_data = [0.6, 0.8, 0.5, 0.7]  # Random feature values
    trade_signal = predict_trade_signals(new_trade_data)
    logging.info(f"Predicted trade signal: {trade_signal}")

    # Monitor profit and trigger automated actions
    monitor_profit_and_transfer(trading_data)




import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Simulated trading data
trading_data = [
    {"trade_id": 1, "profit": 500, "cost": 50, "date": "2025-01-10"},
    {"trade_id": 2, "profit": 700, "cost": 100, "date": "2025-01-11"},
    {"trade_id": 3, "profit": 800, "cost": 120, "date": "2025-01-12"}
]

# Simulated bank transfers
transfer_log = [
    {"date": "2025-01-11", "amount": 200},
    {"date": "2025-01-12", "amount": 300}
]

# Function: Calculate real-time profit
def get_real_time_profit(data):
    total_revenue = sum([trade['profit'] for trade in data])
    total_costs = sum([trade['cost'] for trade in data])
    return total_revenue - total_costs

# Function: Monitor profit and automate transfer
def monitor_profit_and_transfer():
    total_profit = get_real_time_profit(trading_data)
    total_transferred = sum([log["amount"] for log in transfer_log])
    remaining_profit = total_profit - total_transferred

    return total_profit, total_transferred, remaining_profit

# Main Streamlit Dashboard
st.title("Trading Automation System Dashboard")

# Calculate profits
total_profit, total_transferred, remaining_profit = monitor_profit_and_transfer()

# Display metrics
st.subheader("Profit Overview")
st.metric(label="Total Profit", value=f"${total_profit}")
st.metric(label="Total Transferred to Bank", value=f"${total_transferred}")
st.metric(label="Remaining Profit", value=f"${remaining_profit}")

# Trading Trends
st.subheader("Profit Trends Over Time")
trading_df = pd.DataFrame(trading_data)
trading_df['date'] = pd.to_datetime(trading_df['date'])
st.line_chart(trading_df.set_index('date')['profit'])

# Bar Chart: Profit vs. Costs
st.subheader("Profit vs. Costs")
bar_data = trading_df[['date', 'profit', 'cost']].set_index('date')
st.bar_chart(bar_data)

# Pie Chart: Profit Distribution
st.subheader("Profit Distribution")
total_costs = trading_df['cost'].sum()
pie_data = pd.Series([total_profit, total_costs], index=['Profit', 'Costs'])
st.write(pie_data.plot.pie(autopct='%1.1f%%', title="Profit vs. Costs Distribution", figsize=(5, 5)))
st.pyplot()

# Cumulative Profit Trend
st.subheader("Cumulative Profit Trend")
trading_df['cumulative_profit'] = trading_df['profit'].cumsum()
st.area_chart(trading_df.set_index('date')['cumulative_profit'])

# Transaction Logs
st.subheader("Bank Transfer Logs")
transfer_df = pd.DataFrame(transfer_log)
transfer_df['date'] = pd.to_datetime(transfer_df['date'])
st.write(transfer_df)

# Recommendations
st.subheader("Recommendations")
if remaining_profit > 1000:
    st.success("Consider transferring more profits to the bank to optimize cash flow.")
else:
    st.warning("Monitor trading activities to increase profitability before transferring.")

# Manual Transfer Trigger (Optional)
if st.button("Trigger Manual Profit Transfer"):
    st.info("Profit transfer triggered. (Simulation)")
    logging.info("Manual profit transfer triggered.")
