import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.trades as trades

# Initialize Flask App
app = Flask(_name_)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "sqlite:///trading.db")
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "supersecretkey")
db = SQLAlchemy(app)

# Logging Setup
logging.basicConfig(filename='trading_app.log', level=logging.INFO)

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Flask-Limiter Setup (Rate Limiting)
limiter = Limiter(app, key_func=get_remote_address, default_limits=["5 per minute"])

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


def send_email(subject, body, to_email):
    """ Securely send email notifications. """
    from_email = os.getenv("EMAIL_USER")
    from_password = os.getenv("EMAIL_PASS")
    
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


@app.route('/trade', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def trade():
    """ Handles trade execution securely. """
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

    # Store trade in database
    trade = Trade(user_id=current_user.id, pair=pair, action=action, units=units)
    db.session.add(trade)
    db.session.commit()

    # Execute trade on OANDA
    trade_exec = trades.TradeCreate(accountID=OANDA_ACCOUNT_ID, data={
        "order": {
            "units": str(units),
            "instrument": pair,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    })
    api.request(trade_exec)

    logging.info(f"Trade executed: Pair={pair}, Action={action}, Units={units}")
    return redirect(url_for('home'))


### APP START ###
if _name_ == '_main_':
    db.create_all()  # Ensure database is initialized
    app.run(ssl_context=('cert.pem', 'key.pem'))  # Run with HTTPS
    @app.route('/trade', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def trade():
    """ Handles trade execution securely with Stop Loss and Take Profit. """
    pair = request.form['pair']
    action = request.form['action']
    units = request.form['units']
    stop_loss = request.form.get('stop_loss', None)
    take_profit = request.form.get('take_profit', None)
    
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

    if stop_loss:
        try:
            stop_loss = float(stop_loss)
        except ValueError:
            return "Invalid Stop Loss value", 400

    if take_profit:
        try:
            take_profit = float(take_profit)
        except ValueError:
            return "Invalid Take Profit value", 400

    # Check user role
    if current_user.role != "trader":
        return "Access Denied", 403

    # Store trade in database
    trade = Trade(user_id=current_user.id, pair=pair, action=action, units=units)
    db.session.add(trade)
    db.session.commit()
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_required, current_user
import time
import threading
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pair = db.Column(db.String(10))
    action = db.Column(db.String(5))
    units = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class UserPreferences(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    preferred_pair = db.Column(db.String(10), nullable=False)
    update_frequency = db.Column(db.Integer, default=30)  # Default update frequency (in seconds)

# Function to get market data (simplified)
def get_forex_data(pair):
    # Mocked data for demonstration purposes
    return {'candles': [{'mid': {'c': 1.2}}]}  # Simplified API response

@app.route('/')
@login_required
def dashboard():
    """ Displays the trading dashboard. """
    return render_template('dashboard.html')

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

@app.route('/trading-history')
@login_required
def trading_history():
    trades = Trade.query.filter_by(user_id=current_user.id).all()
    return render_template('trading_history.html', trades=trades)

# Market data updates and trade execution (simplified for demonstration)
def market_data_update():
    while True:
        pair = "EUR/USD"
        data = get_forex_data(pair)
        price = data['candles'][-1]['mid']['c']
        socketio.emit('market_data', {'price': price, 'time': datetime.utcnow().strftime('%H:%M:%S')}, broadcast=True)
        time.sleep(30)

if __name__ == "__main__":
    app.run(debug=True)

    # Construct trade data for OANDA API
    trade_data = {
        "order": {
            "units": str(units),
            "instrument": pair,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }

    # Add SL & TP if provided
    if stop_loss:
        trade_data["order"]["stopLossOnFill"] = {"price": str(stop_loss)}
    if take_profit:
        trade_data["order"]["takeProfitOnFill"] = {"price": str(take_profit)}

    # Execute trade on OANDA
    trade_exec = trades.TradeCreate(accountID=OANDA_ACCOUNT_ID, data=trade_data)
    api.request(trade_exec)

    logging.info(f"Trade executed: Pair={pair}, Action={action}, Units={units}, SL={stop_loss}, TP={take_profit}")
    return redirect(url_for('home'))
    
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

@app.route('/trade', methods=['POST'])
@login_required
def trade():
    """ Executes trades with automatic position sizing. """
    pair = request.form['pair']
    action = request.form['action']
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
    <!-- dashboard.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.0.1/socket.io.min.js"></script>
</head>
<body>
    <h1>Trading Dashboard</h1>
    <div>
        <h2>Real-Time Forex Chart</h2>
        <canvas id="priceChart" width="400" height="200"></canvas>
    </div>
</body>
</html>

