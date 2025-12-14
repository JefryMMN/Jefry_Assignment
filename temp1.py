"""
Assignment 1: Temperature Analysis
SUPER BASIC - Only What You Need!
"""

import pandas as pd
import matplotlib.pyplot as plt


wa = pd.read_csv('4180571.csv')
ma = pd.read_csv('4185951.csv')
tx = pd.read_csv('4180582.csv')
il = pd.read_csv('4180585.csv')
wi = pd.read_csv('4180586.csv')



print("\n=== STATISTICS ===\n")

print("WASHINGTON:")
print(f"  Mean: {wa['TAVG'].mean():.1f}°F")
print(f"  Min: {wa['TAVG'].min():.1f}°F")
print(f"  Max: {wa['TAVG'].max():.1f}°F")

print("\nMASSACHUSETTS:")
print(f"  Mean: {ma['TAVG'].mean():.1f}°F")
print(f"  Min: {ma['TAVG'].min():.1f}°F")
print(f"  Max: {ma['TAVG'].max():.1f}°F")

print("\nTEXAS:")
print(f"  Mean: {tx['TAVG'].mean():.1f}°F")
print(f"  Min: {tx['TAVG'].min():.1f}°F")
print(f"  Max: {tx['TAVG'].max():.1f}°F")

print("\nILLINOIS:")
print(f"  Mean: {il['TAVG'].mean():.1f}°F")
print(f"  Min: {il['TAVG'].min():.1f}°F")
print(f"  Max: {il['TAVG'].max():.1f}°F")

print("\nWISCONSIN:")
print(f"  Mean: {wi['TAVG'].mean():.1f}°F")
print(f"  Min: {wi['TAVG'].min():.1f}°F")
print(f"  Max: {wi['TAVG'].max():.1f}°F")


print("\n=== CORRELATION ===\n")

correlation = wa['TAVG'].corr(ma['TAVG'])
print(f"Washington vs Massachusetts: {correlation:.3f}")

if correlation > 0.7:
    print("→ HIGH! They move together!")
elif correlation < 0.3:
    print("→ LOW! They move separately!")
else:
    print("→ MEDIUM")


print("\n=== CREATING CHARTS ===\n")

plt.figure(figsize=(10, 6))
plt.hist(wa['TAVG'], bins=30, color='blue', edgecolor='black')
plt.title('Washington Temperature Distribution')
plt.xlabel('Temperature (°F)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(ma['TAVG'], bins=30, color='green', edgecolor='black')
plt.title('Massachusetts Temperature Distribution')
plt.xlabel('Temperature (°F)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(tx['TAVG'], bins=30, color='red', edgecolor='black')
plt.title('Texas Temperature Distribution')
plt.xlabel('Temperature (°F)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(il['TAVG'], bins=30, color='orange', edgecolor='black')
plt.title('Illinois Temperature Distribution')
plt.xlabel('Temperature (°F)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(wi['TAVG'], bins=30, color='purple', edgecolor='black')
plt.title('Wisconsin Temperature Distribution')
plt.xlabel('Temperature (°F)')
plt.ylabel('Frequency')
plt.show()

print("\n=== CORRELATION HEATMAP ===\n")

# Prepare data: Merge all states by date
wa['DATE'] = pd.to_datetime(wa['DATE'])
ma['DATE'] = pd.to_datetime(ma['DATE'])
tx['DATE'] = pd.to_datetime(tx['DATE'])
il['DATE'] = pd.to_datetime(il['DATE'])
wi['DATE'] = pd.to_datetime(wi['DATE'])

merged = wa[['DATE', 'TAVG']].rename(columns={'TAVG': 'Washington'})
merged = merged.merge(ma[['DATE', 'TAVG']].rename(columns={'TAVG': 'Massachusetts'}), on='DATE')
merged = merged.merge(tx[['DATE', 'TAVG']].rename(columns={'TAVG': 'Texas'}), on='DATE')
merged = merged.merge(il[['DATE', 'TAVG']].rename(columns={'TAVG': 'Illinois'}), on='DATE')
merged = merged.merge(wi[['DATE', 'TAVG']].rename(columns={'TAVG': 'Wisconsin'}), on='DATE')


correlation_table = merged[['Washington', 'Massachusetts', 'Texas', 'Illinois', 'Wisconsin']].corr()

print("Correlation Table:")
print(correlation_table.round(3))

import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_table, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, vmin=-1, vmax=1, square=True, linewidths=1)
plt.title('Correlation Heatmap: Temperature Between States', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n=== MONTHLY CORRELATION (WA vs MA) ===\n")

merged['Month'] = pd.to_datetime(merged['DATE']).dt.month

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_correlations = []

for month_num in range(1, 13):

    month_data = merged[merged['Month'] == month_num]
    

    corr = month_data['Washington'].corr(month_data['Massachusetts'])
    monthly_correlations.append(corr)

    if corr >= 0.7:
        level = "HIGH"
    elif corr <= 0.3:
        level = "LOW"
    else:
        level = "MEDIUM"
    
    print(f"{months[month_num-1]}: {corr:.3f} ({level})")

plt.figure(figsize=(12, 6))
plt.bar(months, monthly_correlations, color='steelblue', edgecolor='black')
plt.axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='High (≥0.7)')
plt.axhline(y=0.3, color='red', linestyle='--', linewidth=2, label='Low (≤0.3)')
plt.title('Monthly Correlation: Washington vs Massachusetts', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Correlation Coefficient')
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


print("\n=== SEASONAL CORRELATION (WA vs MA) ===\n")

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

merged['Season'] = merged['Month'].apply(get_season)


seasons = ['Winter', 'Spring', 'Summer', 'Fall']
seasonal_correlations = []

for season in seasons:
    season_data = merged[merged['Season'] == season]
    corr = season_data['Washington'].corr(season_data['Massachusetts'])
    seasonal_correlations.append(corr)
    
    if corr >= 0.7:
        level = "STRONG"
    elif corr >= 0.4:
        level = "MODERATE"
    else:
        level = "WEAK"
    
    print(f"{season}: {corr:.3f} ({level})")

plt.figure(figsize=(10, 6))
plt.bar(seasons, seasonal_correlations, color=['cyan', 'lightgreen', 'orange', 'brown'], edgecolor='black')
plt.axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='Strong (≥0.7)')
plt.title('Seasonal Correlation: Washington vs Massachusetts', fontsize=14, fontweight='bold')
plt.xlabel('Season')
plt.ylabel('Correlation Coefficient')
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
