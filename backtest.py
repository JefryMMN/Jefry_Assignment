import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

SYMBOLS = ['SPY', 'AAPL', 'NVDA', 'MSFT', 'IBM', 'DIS', 'HOOD']
INITIAL_CAPITAL = 1000
START_DATE = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

data = {}
for symbol in SYMBOLS:
    df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    data[symbol] = df

def calculate_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['VolMA'] = df['Volume'].rolling(window=20).mean()
    
    return df

def generate_signals(df):
    df['Buy'] = 0
    df['Sell'] = 0
    
    for i in range(1, len(df)):
        if (pd.notna(df['SMA20'].iloc[i]) and pd.notna(df['SMA50'].iloc[i]) and
            pd.notna(df['MACD'].iloc[i]) and pd.notna(df['Signal'].iloc[i]) and
            pd.notna(df['VolMA'].iloc[i])):
            
            sma_cross_up = (df['SMA20'].iloc[i] > df['SMA50'].iloc[i] and 
                           df['SMA20'].iloc[i-1] <= df['SMA50'].iloc[i-1])
            
            macd_cross_up = (df['MACD'].iloc[i] > df['Signal'].iloc[i] and
                            df['MACD'].iloc[i-1] <= df['Signal'].iloc[i-1])
            
            volume_high = df['Volume'].iloc[i] > df['VolMA'].iloc[i]
            
            sma_bullish = df['SMA20'].iloc[i] > df['SMA50'].iloc[i]
            macd_bullish = df['MACD'].iloc[i] > df['Signal'].iloc[i]
            
            if sma_cross_up and macd_bullish and volume_high:
                df.loc[df.index[i], 'Buy'] = 1
            
            sma_cross_down = (df['SMA20'].iloc[i] < df['SMA50'].iloc[i] and 
                             df['SMA20'].iloc[i-1] >= df['SMA50'].iloc[i-1])
            
            macd_cross_down = (df['MACD'].iloc[i] < df['Signal'].iloc[i] and
                              df['MACD'].iloc[i-1] >= df['Signal'].iloc[i-1])
            
            if sma_cross_down and macd_cross_down:
                df.loc[df.index[i], 'Sell'] = 1
    
    return df

def backtest(df, symbol):
    capital = INITIAL_CAPITAL
    position = 0
    shares = 0
    trades = []
    equity_curve = []
    
    for i in range(len(df)):
        equity = capital + (shares * df['Close'].iloc[i] if shares > 0 else 0)
        equity_curve.append(equity)
        
        if df['Buy'].iloc[i] == 1 and position == 0:
            shares = capital / df['Close'].iloc[i]
            position = 1
            trades.append({
                'Date': df.index[i],
                'Type': 'BUY',
                'Price': df['Close'].iloc[i],
                'Shares': shares,
                'Capital': capital
            })
            capital = 0
        
        elif df['Sell'].iloc[i] == 1 and position == 1:
            capital = shares * df['Close'].iloc[i]
            position = 0
            trades.append({
                'Date': df.index[i],
                'Type': 'SELL',
                'Price': df['Close'].iloc[i],
                'Shares': shares,
                'Capital': capital
            })
            shares = 0
    
    if position == 1:
        capital = shares * df['Close'].iloc[-1]
        shares = 0
    
    final_capital = capital
    df['Equity'] = equity_curve
    
    return df, trades, final_capital

def calculate_stats(trades, final_capital, df):
    if len(trades) < 2:
        return {
            'Total Trades': 0,
            'Win Rate': 0,
            'Net Gain %': ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100,
            'Sharpe Ratio': 0,
            'Max Drawdown %': 0
        }
    
    wins = 0
    for i in range(1, len(trades), 2):
        if i < len(trades) and trades[i]['Capital'] > trades[i-1]['Capital']:
            wins += 1
    
    win_rate = (wins / (len(trades) // 2)) * 100 if len(trades) >= 2 else 0
    
    returns = df['Equity'].pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    cummax = df['Equity'].cummax()
    drawdown = (df['Equity'] - cummax) / cummax * 100
    max_dd = drawdown.min()
    
    return {
        'Total Trades': len(trades) // 2,
        'Win Rate': win_rate,
        'Net Gain %': ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100,
        'Sharpe Ratio': sharpe,
        'Max Drawdown %': max_dd
    }

results = {}

for symbol in SYMBOLS:
    df = data[symbol].copy()
    df = calculate_indicators(df)
    df = generate_signals(df)
    df, trades, final_capital = backtest(df, symbol)
    stats = calculate_stats(trades, final_capital, df)
    
    results[symbol] = {
        'df': df,
        'trades': trades,
        'final_capital': final_capital,
        'stats': stats
    }

print("BACKTEST RESULTS - 10 YEAR PERIOD")
print(f"\n{'Symbol':<10} {'Trades':<10} {'Win Rate':<12} {'Net Gain %':<15} {'Sharpe':<10} {'Max DD %':<10}")
for symbol in SYMBOLS:
    stats = results[symbol]['stats']
    num_trades = len(results[symbol]['trades'])
    print(f"{symbol:<10} {num_trades:<10} {stats['Win Rate']:<12.2f} {stats['Net Gain %']:<15.2f} {stats['Sharpe Ratio']:<10.2f} {stats['Max Drawdown %']:<10.2f}")


print("5-YEAR PERFORMANCE (Last 5 Years)")

print(f"\n{'Symbol':<10} {'Trades':<10} {'Net Gain %':<15} {'Sharpe':<10} {'Max DD %':<10}")

results_5yr = {}
five_years_ago = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

for symbol in SYMBOLS:
    df_5yr = results[symbol]['df'][results[symbol]['df'].index >= five_years_ago].copy()
    
    if len(df_5yr) > 0:
        initial_equity = df_5yr['Equity'].iloc[0]
        final_equity = df_5yr['Equity'].iloc[-1]
        net_gain_5yr = ((final_equity - initial_equity) / initial_equity) * 100
        
        trades_5yr = [t for t in results[symbol]['trades'] if t['Date'] >= pd.Timestamp(five_years_ago)]
        
        returns_5yr = df_5yr['Equity'].pct_change().dropna()
        sharpe_5yr = (returns_5yr.mean() / returns_5yr.std()) * np.sqrt(252) if returns_5yr.std() > 0 else 0
        
        cummax_5yr = df_5yr['Equity'].cummax()
        drawdown_5yr = (df_5yr['Equity'] - cummax_5yr) / cummax_5yr * 100
        max_dd_5yr = drawdown_5yr.min()
        
        print(f"{symbol:<10} {len(trades_5yr):<10} {net_gain_5yr:<15.2f} {sharpe_5yr:<10.2f} {max_dd_5yr:<10.2f}")

fig, axes = plt.subplots(len(SYMBOLS), 2, figsize=(15, 4*len(SYMBOLS)))

for idx, symbol in enumerate(SYMBOLS):
    df = results[symbol]['df']
    
    ax1 = axes[idx, 0] if len(SYMBOLS) > 1 else axes[0]
    ax1.plot(df.index, df['Close'], label='Close', linewidth=1)
    ax1.plot(df.index, df['SMA20'], label='SMA20', alpha=0.7)
    ax1.plot(df.index, df['SMA50'], label='SMA50', alpha=0.7)
    
    buy_signals = df[df['Buy'] == 1]
    sell_signals = df[df['Sell'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy', zorder=5)
    ax1.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=100, label='Sell', zorder=5)
    
    ax1.set_title(f'{symbol} - Price & Signals')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[idx, 1] if len(SYMBOLS) > 1 else axes[1]
    ax2.plot(df.index, df['Equity'], label='Equity', linewidth=2)
    ax2.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', label='Initial Capital')
    ax2.set_title(f'{symbol} - Equity Curve')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Equity ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(10, 6))
symbols_list = list(results.keys())
returns = [results[s]['stats']['Net Gain %'] for s in symbols_list]
colors = ['green' if r > 0 else 'red' for r in returns]
ax.bar(symbols_list, returns, color=colors)
ax.set_title('Annualized Returns by Stock')
ax.set_xlabel('Symbol')
ax.set_ylabel('Net Gain (%)')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('returns_comparison.png', dpi=150, bbox_inches='tight')

plt.show()