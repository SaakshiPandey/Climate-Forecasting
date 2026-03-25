"""
Simplified Model Visualization - Actual vs Hybrid Model
Temperature Forecasting Results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for clean plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Create directories
os.makedirs('data/processed/plots', exist_ok=True)

print("="*80)
print("HYBRID MODEL VISUALIZATION - Actual vs Hybrid Predictions")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading prediction data...")

# Load LightGBM predictions (to get actual values and dates)
lgbm_test = pd.read_csv('data/processed/lgbm_test_preds.csv')
print("✅ LightGBM test data loaded")

# Load Hybrid predictions
try:
    ridge_test = pd.read_csv('data/processed/ridge_ensemble_predictions.csv')
    hybrid_col = 'ridge_ensemble_pred'
    print("✅ Ridge Ensemble predictions loaded")
except:
    try:
        ridge_test = pd.read_csv('data/processed/hybrid_predictions_final.csv')
        hybrid_col = 'hybrid_pred'
        print("✅ Hybrid predictions loaded")
    except:
        ridge_test = pd.read_csv('data/processed/hybrid_predictions.csv')
        hybrid_col = 'hybrid_pred'
        print("✅ Hybrid predictions loaded")

# Prepare dataframe
test_df = lgbm_test.copy()
test_df['Date'] = pd.to_datetime(test_df['Date'])
ridge_test['Date'] = pd.to_datetime(ridge_test['Date'])
test_df = test_df.merge(ridge_test[['Date', 'City', hybrid_col]], on=['Date', 'City'], how='inner')
test_df = test_df.rename(columns={hybrid_col: 'hybrid_pred'})

print(f"\n✅ Test data: {len(test_df)} samples")
print(f"   Date range: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
print(f"   Temperature range: {test_df['y_true'].min():.2f}°C to {test_df['y_true'].max():.2f}°C")

# ============================================================================
# 2. CALCULATE METRICS
# ============================================================================
print("\n2. Calculating performance metrics...")

rmse = np.sqrt(mean_squared_error(test_df['y_true'], test_df['hybrid_pred']))
mae = mean_absolute_error(test_df['y_true'], test_df['hybrid_pred'])
r2 = r2_score(test_df['y_true'], test_df['hybrid_pred'])

print(f"\n   Hybrid Model Performance:")
print(f"   RMSE: {rmse:.4f}°C")
print(f"   MAE:  {mae:.4f}°C")
print(f"   R²:   {r2:.4f}")

# ============================================================================
# 3. PLOT 1: Time Series - Actual vs Hybrid (First 200 Days)
# ============================================================================
print("\n3. Creating Plot 1: Time Series Comparison...")

n_samples = min(200, len(test_df))
plot_df = test_df.head(n_samples).copy()
plot_df['index'] = range(len(plot_df))

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(plot_df['index'], plot_df['y_true'], 'k-', linewidth=2, label='Actual Temperature', alpha=0.9)
ax.plot(plot_df['index'], plot_df['hybrid_pred'], 'r-', linewidth=2, label='Hybrid Model Prediction', alpha=0.8)

ax.set_xlabel('Time Steps (Days)')
ax.set_ylabel('Temperature (°C)')
ax.set_title(f'Hybrid Model Predictions vs Actual Temperature\n(First {n_samples} Days)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/plots/01_actual_vs_hybrid_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 01_actual_vs_hybrid_timeseries.png")

# ============================================================================
# 4. PLOT 2: Scatter Plot - Predicted vs Actual
# ============================================================================
print("\n4. Creating Plot 2: Predicted vs Actual Scatter...")

fig, ax = plt.subplots(figsize=(10, 10))

# Scatter plot
ax.scatter(test_df['y_true'], test_df['hybrid_pred'], alpha=0.5, s=20, c='red', edgecolors='black', linewidth=0.5)

# Perfect prediction line
min_val = min(test_df['y_true'].min(), test_df['hybrid_pred'].min())
max_val = max(test_df['y_true'].max(), test_df['hybrid_pred'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

# Add metrics text
ax.text(0.05, 0.95, f'RMSE: {rmse:.3f}°C\nMAE: {mae:.3f}°C\nR²: {r2:.3f}', 
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

ax.set_xlabel('Actual Temperature (°C)')
ax.set_ylabel('Hybrid Model Prediction (°C)')
ax.set_title('Hybrid Model: Predicted vs Actual Temperature')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('data/processed/plots/02_hybrid_scatter_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 02_hybrid_scatter_plot.png")

# ============================================================================
# 5. PLOT 3: Error Distribution
# ============================================================================
print("\n5. Creating Plot 3: Error Distribution...")

errors = test_df['y_true'] - test_df['hybrid_pred']

fig, ax = plt.subplots(figsize=(12, 6))

# Histogram
n, bins, patches = ax.hist(errors, bins=50, alpha=0.7, color='red', edgecolor='black', density=True)

# Add normal curve
mu = errors.mean()
sigma = errors.std()
x = np.linspace(errors.min(), errors.max(), 100)
y = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)
ax.plot(x, y, 'k-', linewidth=2, label=f'Normal Distribution\nμ={mu:.3f}°C, σ={sigma:.3f}°C')

ax.axvline(x=0, color='k', linestyle='--', linewidth=1, label='Zero Error')
ax.axvline(x=mu, color='r', linestyle='--', linewidth=1, label=f'Mean Error: {mu:.3f}°C')

ax.set_xlabel('Prediction Error (°C)')
ax.set_ylabel('Density')
ax.set_title('Hybrid Model Error Distribution')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Add statistics text
stats_text = f'Mean Error: {mu:.3f}°C\nStd Dev: {sigma:.3f}°C\n95% CI: [{np.percentile(errors, 2.5):.3f}, {np.percentile(errors, 97.5):.3f}]°C'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
        horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('data/processed/plots/03_hybrid_error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 03_hybrid_error_distribution.png")

# ============================================================================
# 6. PLOT 4: Performance by Temperature Range
# ============================================================================
print("\n6. Creating Plot 4: Performance by Temperature Range...")

# Create temperature bins
test_df['temp_range'] = pd.cut(test_df['y_true'], 
                               bins=[-np.inf, 15, 25, 35, np.inf],
                               labels=['Cold (<15°C)', 'Mild (15-25°C)', 
                                      'Warm (25-35°C)', 'Hot (>35°C)'])

# Calculate RMSE for each range
range_rmse = []
range_mae = []
range_samples = []

for temp_range in test_df['temp_range'].unique():
    if pd.notna(temp_range):
        subset = test_df[test_df['temp_range'] == temp_range]
        rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['hybrid_pred']))
        mae = mean_absolute_error(subset['y_true'], subset['hybrid_pred'])
        range_rmse.append(rmse)
        range_mae.append(mae)
        range_samples.append(len(subset))

# Create DataFrame for plotting
range_df = pd.DataFrame({
    'Temperature Range': test_df['temp_range'].unique(),
    'RMSE (°C)': range_rmse,
    'MAE (°C)': range_mae,
    'Samples': range_samples
})

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(range_df))
width = 0.35

bars1 = ax.bar(x - width/2, range_df['RMSE (°C)'], width, label='RMSE', color='red', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, range_df['MAE (°C)'], width, label='MAE', color='orange', alpha=0.7, edgecolor='black')

ax.set_xlabel('Temperature Range')
ax.set_ylabel('Error (°C)')
ax.set_title('Hybrid Model Performance by Temperature Range')
ax.set_xticks(x)
ax.set_xticklabels(range_df['Temperature Range'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add sample count text
for i, (bar, samples) in enumerate(zip(bars1, range_df['Samples'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
            f'n={samples}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('data/processed/plots/04_hybrid_performance_by_temp.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 04_hybrid_performance_by_temp.png")

# ============================================================================
# 7. CREATE SUMMARY REPORT
# ============================================================================
print("\n7. Creating summary report...")

# Calculate additional metrics
max_error = np.max(np.abs(errors))
p95_error = np.percentile(np.abs(errors), 95)
p99_error = np.percentile(np.abs(errors), 99)

summary_df = pd.DataFrame({
    'Metric': ['RMSE (°C)', 'MAE (°C)', 'R² Score', 'Max Error (°C)', 
               '95% Error (°C)', '99% Error (°C)', 'Mean Error (°C)', 
               'Error Std Dev (°C)', 'Total Samples'],
    'Value': [rmse, mae, r2, max_error, p95_error, p99_error, mu, sigma, len(test_df)]
})

print("\n" + "="*80)
print("HYBRID MODEL PERFORMANCE SUMMARY")
print("="*80)
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('data/processed/plots/hybrid_model_summary.csv', index=False)
print("\n✅ Summary saved to data/processed/plots/hybrid_model_summary.csv")

# ============================================================================
# 8. FINAL OUTPUT
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print("\n📁 Files Generated:")
print("   ├── 01_actual_vs_hybrid_timeseries.png  (Time series comparison)")
print("   ├── 02_hybrid_scatter_plot.png          (Predicted vs Actual scatter)")
print("   ├── 03_hybrid_error_distribution.png    (Error histogram)")
print("   ├── 04_hybrid_performance_by_temp.png   (Performance by temperature range)")
print("   └── hybrid_model_summary.csv            (Performance metrics)")

print("\n" + "="*80)
print("✅ DONE!")
print("="*80)