import requests
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def fetch_klines(symbol='BTCUSDT', interval='1h', limit=1000):
    """
    Fetches historical k-line (candlestick) data from Binance.
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an exception for bad responses (4xx or 5xx)
        data = response.json()
        
        # Define the column names as per Binance API documentation
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(data, columns=columns)
        
        # Convert timestamp to datetime and select relevant columns
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        
        # Convert relevant columns to numeric types for calculations
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Return a clean DataFrame
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

    except requests.exceptions.RequestException as e:
        print(f"An error occurred fetching data: {e}")
        return None

# --- NEW FUNCTION: Replaced with loop-based simulation ---
def backtest_ma_crossover(df, 
                          short_window=10, 
                          long_window=50, 
                          trading_fee=0.001, 
                          initial_capital=10000,
                          trade_amount=10000): # <-- Added fixed trade amount
    """
    Backtests a Moving Average Crossover strategy, NOW WITH:
    - Loop-based simulation (required for stops)
    - TRAILING ATR Stop Loss
    """
    if df is None or 'close' not in df.columns:
        print("DataFrame is invalid.")
        return None

    print(f"\nRunning backtest with short={short_window}, long={long_window}, fee={trading_fee}...")
    
    data = df.copy()
    # Use EWM as in your original file
    data['short_ma'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['long_ma'] = data['close'].ewm(span=long_window, adjust=False).mean()
    
    #1. Compute ATR(14) for stop-loss calculation 
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    data['atr'] = tr.rolling(window=14).mean()
    
    # --- 2. Create signals (vectorized) ---
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
    # Use 'crossover_position' to track the raw signal
    data['crossover_position'] = data['signal'].shift(1).fillna(0)
    
    data = data.dropna() # Drop initial rows where MAs or ATR are NaN
    
    if data.empty:
        print("Not enough data to run backtest after processing. Try a longer 'limit'.")
        return None
        
    # 3. Loop-based simulation
    position_qty = 0  
    entry_price = 0
    stop_loss = 0 # This will store our trailing stop
    capital = initial_capital # This is our CASH balance
    portfolio_values = []
    trades = 0
    position_binary_list = [] # For plotting
    
    for i in range(len(data)):
        current_price = data['close'].iloc[i]
        atr = data['atr'].iloc[i]
        
        if position_qty == 0:
            position_binary_list.append(0)
            
            # Check for buy crossover signal
            if data['crossover_position'].iloc[i] == 1:
                if capital >= trade_amount: # Check if we have cash
                    position_qty = trade_amount / current_price
                    entry_price = current_price
                    
                    # --- SET INITIAL STOP LOSS ---
                    stop_loss = current_price - 1.5 * atr 
                    
                    trade_cost = position_qty * current_price
                    trade_fee_cost = trade_cost * trading_fee
                    capital -= (trade_cost + trade_fee_cost) 
                    trades += 1
        else: # We are IN a position
            position_binary_list.append(1)
            
            # --- TRAILING STOP LOGIC ---
            # 1. Calculate the new potential stop loss
            new_stop_loss = current_price - 1.5 * atr
            # 2. Update stop_loss if the new one is HIGHER
            stop_loss = max(stop_loss, new_stop_loss)
            # --- END TRAILING STOP LOGIC ---
            
            # Check if stop-loss is hit
            if current_price <= stop_loss:
                sell_proceeds = position_qty * current_price
                trade_fee_cost = sell_proceeds * trading_fee
                capital += (sell_proceeds - trade_fee_cost)
                position_qty = 0
                trades += 1
            # Check for sell crossover signal
            elif data['crossover_position'].iloc[i] == 0:
                sell_proceeds = position_qty * current_price
                trade_fee_cost = sell_proceeds * trading_fee
                capital += (sell_proceeds - trade_fee_cost)
                position_qty = 0
                trades += 1

        # Update portfolio value (Cash + Asset Value)
        current_portfolio = capital + (position_qty * current_price) 
        portfolio_values.append(current_portfolio)
    
    # Close any open position at the end of the backtest
    if position_qty > 0:
        sell_proceeds = position_qty * current_price
        trade_fee_cost = sell_proceeds * trading_fee
        capital += (sell_proceeds - trade_fee_cost)
        trades += 1
        portfolio_values[-1] = capital # Update last value
    
    data['portfolio_value'] = portfolio_values
    data['position'] = position_binary_list # Add 0/1 position for plotting
    
    # --- 4. Calculate metrics correctly from portfolio value ---
    data['strategy_returns'] = data['portfolio_value'].pct_change().fillna(0)

    # --- Metrics Calculation ---
    total_return = (data['portfolio_value'].iloc[-1] / initial_capital) - 1
    total_trades = trades
    hours_in_year = 365 * 24
    
    mean_return_hourly = data['strategy_returns'].mean()
    annualized_return = (1 + mean_return_hourly) ** hours_in_year - 1

    std_dev_hourly = data['strategy_returns'].std()
    annualized_std_dev = std_dev_hourly * np.sqrt(hours_in_year)
    
    if annualized_std_dev == 0:
        sharpe_ratio = np.inf
    else:
        sharpe_ratio = (mean_return_hourly * hours_in_year) / annualized_std_dev
    
    # 5. Sortino Ratio
    downside_returns = data[data['strategy_returns'] < 0]['strategy_returns']
    downside_deviation_hourly = downside_returns.std()
    
    if downside_deviation_hourly == 0 or pd.isna(downside_deviation_hourly):
        sortino_ratio = np.inf
    else:
        annualized_downside_deviation = downside_deviation_hourly * np.sqrt(hours_in_year)
        sortino_ratio = (mean_return_hourly * hours_in_year) / annualized_downside_deviation
        
    # 6. Calmar Ratio
    data['peak_value'] = data['portfolio_value'].cummax()
    data['drawdown'] = (data['portfolio_value'] - data['peak_value']) / data['peak_value']
    max_drawdown = data['drawdown'].min()
    
    if max_drawdown == 0:
        calmar_ratio = np.inf
    else:
        calmar_ratio = (mean_return_hourly * hours_in_year) / abs(max_drawdown)
        
    print(f"Backtest Complete:")
    print(f"Period: {data['open_time'].iloc[0]} to {data['open_time'].iloc[-1]}")
    print(f"Total Strategy Return: {total_return:.2%}")
    print(f"Total Trades: {total_trades}")
    print("--- Risk & Return Metrics ---")
    print(f"Annualized Return: {mean_return_hourly * hours_in_year:.2%}")
    print(f"Annualized Volatility: {annualized_std_dev:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print("--- Ratios ---")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")
    
    return data

def visualize_backtest(df):
    """
    Visualizes the backtest results, including price, MAs, trade signals, and equity curve.
    (This function is unchanged)
    """
    if df is None:
        print("Cannot visualize an empty DataFrame.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Moving Average Crossover Strategy Backtest', fontsize=16)

    # --- Plot 1: Price, MAs, and Trade Signals ---
    ax1.plot(df['open_time'], df['close'], label='Close Price', color='blue', alpha=0.7)
    ax1.plot(df['open_time'], df['short_ma'], label='Short MA', color='orange', linestyle='--')
    ax1.plot(df['open_time'], df['long_ma'], label='Long MA', color='purple', linestyle='--')

    # This 'position' column is now created by the loop
    buy_signals = df[(df['position'] == 1) & (df['position'].diff() > 0)]
    sell_signals = df[(df['position'] == 0) & (df['position'].diff() < 0)]

    ax1.plot(buy_signals['open_time'], buy_signals['close'], '^', markersize=10, color='green', lw=0, label='Buy Trade')
    ax1.plot(sell_signals['open_time'], sell_signals['close'], 'v', markersize=10, color='red', lw=0, label='Sell Trade')

    ax1.set_title('BTC/USDT Price with Trading Signals')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Equity Curve (using portfolio_value)
    ax2.plot(df['open_time'], df['portfolio_value'], label='Strategy Equity Curve', color='green')
    ax2.set_title('Portfolio Value (Equity Curve)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # Also plot drawdown
    ax3 = ax2.twinx()
    ax3.fill_between(df['open_time'], df['drawdown'] * 100, 0, color='red', alpha=0.3, label='Drawdown')
    ax3.set_ylabel('Drawdown (%)', color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    ax3.legend(loc='lower right')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Main Execution
if _name_ == "_main_":
    print("Fetching historical data for backtesting...")
    historical_data = fetch_klines(
        symbol='BTCUSDT', 
        interval='1h', 
        limit=500) # Using 500 to match your file

    if historical_data is not None:
        # Run the backtest
        backtest_results = backtest_ma_crossover(historical_data,   
                                                 short_window=5, 
                                                 long_window=10, 
                                                 trading_fee=0.0001,
                                                 initial_capital=50000,
                                                 trade_amount=10000 # <-- Added fixed trade amount
                                                 )
        
        # Visualize the results
        if backtest_results is not None:
            visualize_backtest(backtest_results)