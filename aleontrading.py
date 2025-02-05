import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np
from oandapyV20 import API
import oandapyV20.endpoints.trades as trades
import time
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import MetaTrader5 as mt5
import requests
import time
import websocket
import json
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
# Initialize Flask App
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "sqlite:///trading.db")
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "supersecretkey")
db = SQLAlchemy(app)

# Logging Setup
import logging
logging.basicConfig(filename='trading_app.log', level=logging.INFO)

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# OANDA API Setup (Use environment variables)
OANDA_ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
api = API(access_token=OANDA_ACCESS_TOKEN)


### DATABASE MODELS ###
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(50), default="user")  # Can be "admin", "trader"

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    pair = db.Column(db.String(10), nullable=False)
    action = db.Column(db.String(10), nullable=False)
    units = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class UserPreferences(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    preferred_pair = db.Column(db.String(10), nullable=False)
    update_frequency = db.Column(db.Integer, default=30)  # Default update frequency (in seconds)


# Load User
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


### HELPER FUNCTIONS ###
def get_forex_data(pair, count=100):
    """ Fetch forex data from OANDA API. """
    params = {"count": count, "granularity": "M1"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    api.request(r)
    return r.response


def send_email(subject, body, to_email):
    """ Send email notifications. """
    from_email = os.getenv("ontobaeric@gmail.com")
    from_password = os.getenv("aniela@1111")
    
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        logging.error(f"Failed to send email: {e}")


def moving_average_crossover_strategy(pair, data):
    """ Implements a simple Moving Average Crossover Strategy. """
    df = pd.DataFrame(data['candles'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df['SMA_50'] = df['mid']['c'].rolling(window=50).mean()
    df['SMA_200'] = df['mid']['c'].rolling(window=200).mean()
    
    df['signal'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
    df['positions'] = df['signal'].diff()
    return df


def calculate_atr(pair, period=14):
    """ Calculate Average True Range (ATR) for volatility-based Stop Loss. """
    data = get_forex_data(pair, count=period+1)
    df = pd.DataFrame(data['candles'])
    df['high'] = df['mid']['h'].astype(float)
    df['low'] = df['mid']['l'].astype(float)
    df['close'] = df['mid']['c'].astype(float)

    df['tr'] = np.maximum(df['high'] - df['low'], 
                           np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                      abs(df['low'] - df['close'].shift(1))))
    atr = df['tr'].rolling(window=period).mean().iloc[-1]
    return round(atr, 5)  # Return ATR value as stop loss in pips


def calculate_position_size(account_balance, pair, risk_percent=1):
    """ Determine position size based on risk percentage. """
    atr = calculate_atr(pair)
    pip_value = 1  # Assume $1 per pip for a standard lot (adjust if needed)
    
    risk_amount = account_balance * (risk_percent / 100)
    lot_size = risk_amount / (atr * pip_value)

    return round(lot_size, 2)  # Round to 2 decimal places for micro lots


### BACKGROUND TASKS ###

scheduler = BackgroundScheduler()
scheduler.start()

def send_daily_report():
    """ Sends a daily trading report via email. """
    trades = Trade.query.filter(Trade.timestamp >= datetime.utcnow() - timedelta(days=1)).all()
    report = "\n".join([f"Pair: {t.pair}, Action: {t.action}, Units: {t.units}, Time: {t.timestamp}" for t in trades])
    
    send_email("Daily Trading Report", report, "user@example.com")

scheduler.add_job(send_daily_report, 'cron', hour=23, minute=59)


### ROUTES ###

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password", "danger")
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/preferences', methods=['GET', 'POST'])
@login_required
def preferences():
    if request.method == 'POST':
        preferred_pair = request.form['preferred_pair']
        update_frequency = request.form['update_frequency']
        user_prefs = UserPreferences.query.filter_by(user_id=current_user.id).first()
        if user_prefs:
            user_prefs.preferred_pair = preferred_pair
            user_prefs.update_frequency = int(update_frequency)
        else:
            user_prefs = UserPreferences(user_id=current_user.id, preferred_pair=preferred_pair, update_frequency=int(update_frequency))
            db.session.add(user_prefs)
        db.session.commit()
        flash("Preferences updated successfully.", "success")
        return redirect(url_for('preferences'))
    user_prefs = UserPreferences.query.filter_by(user_id=current_user.id).first()
    return render_template('preferences.html', preferences=user_prefs)


@app.route('/trade', methods=['POST'])
@login_required
def trade():
    """ Executes trades with automatic position sizing. """
    pair = request.form['pair']
    action = request.form['action']
    units = request.form['units']
    
    # Validate inputs
    allowed_pairs = ["EUR/USD", "USD/JPY", "GBP/USD"]
    if pair not in allowed_pairs:
        return "Invalid currency pair", 400
    if action not in ["buy", "sell"]:
        return "Invalid action", 400
    try:
        units = int(units)
        if units <= 0:
            return "Invalid unit amount", 400
    except ValueError:
        return "Units must be a number", 400

    # Check user role
    if current_user.role != "trader":
        return "Access Denied", 403

    account_balance = 200  # Example balance (could be dynamically fetched)

    # Calculate position size
    lot_size = calculate_position_size(account_balance, pair)

    # Calculate SL & TP
    atr = calculate_atr(pair)
    stop_loss = round(float(get_forex_data(pair, count=1)['candles'][-1]['mid']['c']) - atr, 5)
    take_profit = round(stop_loss + (atr * 2), 5)  # 1:2 RRR

    # Execute trade
    trade_data = {
        "order": {
            "units": str(lot_size),
            "instrument": pair,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {"price": str(stop_loss)},
            "takeProfitOnFill": {"price": str(take_profit)}
        }
    }

    trade_exec = trades.TradeCreate(accountID=OANDA_ACCOUNT_ID, data=trade_data)
    api.request(trade_exec)

    return f"Trade placed: {pair} | {action} | {lot_size} lots | SL: {stop_loss} | TP: {take_profit}"
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    pair = db.Column(db.String(10), nullable=False)
    action = db.Column(db.String(10), nullable=False)
    units = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    profit_loss = db.Column(db.Float, default=0.0)  # Profit or loss
    saved_amount = db.Column(db.Float, default=0.0)  # Saved profit amount

class Savings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    total_saved = db.Column(db.Float, default=0.0)  # Total saved amount

def close_trade(trade_id, closing_price):
    """ Calculate profit/loss and save a percentage into savings. """
    trade = Trade.query.get(trade_id)
    
    if trade:
        # Assuming get_forex_data retrieves the entry price
        entry_price = float(get_forex_data(trade.pair, count=1)['candles'][-1]['mid']['c'])
        
        # Simplified profit calculation (positive or negative based on the action)
        if trade.action == 'buy':
            trade.profit_loss = (closing_price - entry_price) * trade.units
        elif trade.action == 'sell':
            trade.profit_loss = (entry_price - closing_price) * trade.units

        # Save 10% of the profit
        savings_percentage = 10
        saved_amount = (savings_percentage / 100) * trade.profit_loss
        trade.saved_amount = saved_amount

        # Update the savings table for the user
        savings = Savings.query.filter_by(user_id=trade.user_id).first()
        if not savings:
            savings = Savings(user_id=trade.user_id, total_saved=0.0)
            db.session.add(savings)

        savings.total_saved += saved_amount

        # Commit the changes to the database
        db.session.commit()

        return f"Trade closed! Profit: ${trade.profit_loss:.2f}, Saved: ${saved_amount:.2f}"
    
    return "Trade not found."
from flask import request

@app.route('/withdraw', methods=['POST'])
@login_required
def withdraw():
    """Handle user withdrawal from savings."""
    user_savings = Savings.query.filter_by(user_id=current_user.id).first()
    if user_savings:
        amount = float(request.form['amount'])
        
        # Check if the user has enough savings
        if amount <= user_savings.total_saved:
            user_savings.total_saved -= amount
            db.session.commit()
            return f"Successfully withdrew ${amount:.2f}. Remaining savings: ${user_savings.total_saved:.2f}"
        else:
            return "Insufficient savings balance", 400

    return "No savings found for this user.", 404
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import time
import websocket
import json
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ===========================
# Step 1: Connect to MT5
# ===========================
def connect_mt5(login=12345678, password="your_password", server="Your_Broker_Server"):
    if not mt5.initialize():
        print("MT5 connection failed!")
        mt5.shutdown()
        return False

    if not mt5.login(login, password, server):
        print("Login failed")
        mt5.shutdown()
        return False

    print("Connected to MT5!")
    return True

# ===========================
# Step 2: WebSocket Real-Time Data
# ===========================
def on_message(ws, message):
    data = json.loads(message)
    price = data["price"]
    symbol = data["symbol"]
    print(f"Real-time {symbol} Price: {price}")

def start_websocket():
    ws = websocket.WebSocketApp("wss://your-websocket-url", on_message=on_message)
    ws.run_forever()

# ===========================
# Step 3: Execute Trades with Trailing Stop-Loss
# ===========================
def execute_trade(symbol, lot_size, direction, stop_loss=None, take_profit=None):
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if direction == "BUY" else mt5.symbol_info_tick(symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 20,
        "magic": 123456,
        "comment": "AutoTrade",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    order = mt5.order_send(request)
    if order.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Trade executed: {direction} {lot_size} {symbol} at {price}")
        return order.order
    else:
        print(f"Trade failed: {order.comment}")
        return None

# ===========================
# Step 4: Trailing Stop-Loss Logic
# ===========================
def update_trailing_stop(symbol, trade_id, trailing_pips=10):
    position = mt5.positions_get(ticket=trade_id)
    if not position:
        print("Trade not found!")
        return

    price = mt5.symbol_info_tick(symbol).bid if position[0].type == 1 else mt5.symbol_info_tick(symbol).ask
    new_stop_loss = price - (trailing_pips * 0.0001) if position[0].type == 0 else price + (trailing_pips * 0.0001)

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": trade_id,
        "sl": new_stop_loss,
    }
    
    order = mt5.order_send(request)
    if order.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Trailing Stop Updated for Trade {trade_id}")
    else:
        print(f"Trailing Stop Update Failed: {order.comment}")

# ===========================
# Step 5: AI Trading with XGBoost & LSTM
# ===========================
def train_xgboost_model():
    df = pd.read_csv("forex_data.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    model = XGBClassifier()
    model.fit(X, y)
    return model

def train_lstm_model():
    df = pd.read_csv("forex_data.csv")
    data = df["close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(data)):
        X.append(data[i-60:i])
        y.append(data[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=5, batch_size=32)
    return model, scaler

# ===========================
# Step 6: AI Trade Decision
# ===========================
xgb_model = train_xgboost_model()
lstm_model, scaler = train_lstm_model()

def ai_trade_decision(symbol="EURUSD"):
    latest_data = np.random.rand(1, 10)  # Replace with real forex features
    xgb_pred = xgb_model.predict(latest_data)

    latest_prices = np.random.rand(60, 1)  # Replace with real-time data
    latest_prices_scaled = scaler.transform(latest_prices)
    lstm_pred = lstm_model.predict(latest_prices_scaled.reshape(1, 60, 1))

    if xgb_pred == 1 and lstm_pred > 0.5:
        trade_id = execute_trade(symbol, 0.1, "BUY")
    elif xgb_pred == 0 and lstm_pred < 0.5:
        trade_id = execute_trade(symbol, 0.1, "SELL")

    if trade_id:
        update_trailing_stop(symbol, trade_id)

# ===========================
# Step 7: Run the Trading Bot
# ===========================
if connect_mt5():
    ws_thread = threading.Thread(target=start_websocket)
    ws_thread.start()

    while True:
        ai_trade_decision()  # Execute AI-based trade
        time.sleep(5)  # Wait before next cycle

