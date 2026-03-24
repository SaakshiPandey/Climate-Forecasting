"""
Model Comparison Visualization Script
Actual vs LightGBM vs GRU vs Hybrid Ensemble
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Create directories for saving plots
os.makedirs('data/processed/plots', exist_ok=True)

print("="*80)
print("TEMPERATURE FORECASTING: MODEL COMPARISON")
print("LightGBM vs GRU vs Hybrid Ensemble")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading prediction files...")

# Load LightGBM predictions
lgbm_val = pd.read_csv('data/processed/lgbm_val_preds.csv')
lgbm_test = pd.read_csv('data/processed/lgbm_test_preds.csv')

# Load GRU predictions
gru_val = pd.read_csv('data/processed/gru_val_preds.csv')
gru_test = pd.read_csv('data/processed/gru_test_preds.csv')

# Merge to create complete validation dataset
val_df = lgbm_val.merge(
    gru_val[['Date', 'City', 'gru_pred']], 
    on=['Date', 'City']
)

# Merge to create complete test dataset
test_df = lgbm_test.merge(
    gru_test[['Date', 'City', 'gru_pred']], 
    on=['Date', 'City']
)

# Convert dates
val_df['Date'] = pd.to_datetime(val_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

print(f"✓ Validation samples: {len(val_df)}")
print(f"✓ Test samples: {len(test_df)}")

# ============================================================================
# 2. GENERATE HYBRID PREDICTIONS (RIDGE STACKING)
# ============================================================================
print("\n2. Generating hybrid predictions using Ridge stacking...")

# Create features
val_df['pred_diff'] = val_df['lgbm_pred'] - val_df['gru_pred']
test_df['pred_diff'] = test_df['lgbm_pred'] - test_df['gru_pred']

# Features for stacking
features = ['lgbm_pred', 'gru_pred', 'pred_diff']
X_train = val_df[features].values
y_train = val_df['y_true'].values
X_test = test_df[features].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge model
ridge = Ridge(alpha=0.01)
ridge.fit(X_train_scaled, y_train)

# Generate hybrid predictions
hybrid_pred = ridge.predict(X_test_scaled)
test_df['hybrid_pred'] = hybrid_pred

print(f"✓ Hybrid predictions generated")
print(f"\nRidge Stacking Coefficients:")
for feat, coef in zip(features, ridge.coef_):
    print(f"  {feat:12s}: {coef:8.4f}")
print(f"  {'intercept':12s}: {ridge.intercept_:8.4f}")

# ============================================================================
# 3. CALCULATE PERFORMANCE METRICS
# ============================================================================
print("\n3. Calculating performance metrics...")

models = {
    'LightGBM': test_df['lgbm_pred'],
    'GRU': test_df['gru_pred'],
    'Hybrid (Ridge)': test_df['hybrid_pred']
}

metrics_data = []
for name, predictions in models.items():
    rmse = np.sqrt(mean_squared_error(test_df['y_true'], predictions))
    mae = mean_absolute_error(test_df['y_true'], predictions)
    r2 = r2_score(test_df['y_true'], predictions)
    max_error = np.max(np.abs(test_df['y_true'] - predictions))
    mean_error = np.mean(test_df['y_true'] - predictions)
    
    metrics_data.append({
        'Model': name,
        'RMSE (°C)': rmse,
        'MAE (°C)': mae,
        'R² Score': r2,
        'Max Error (°C)': max_error,
        'Bias (°C)': mean_error
    })

metrics_df = pd.DataFrame(metrics_data)

print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)
print(metrics_df.to_string(index=False))

# Save metrics
metrics_df.to_csv('data/processed/plots/model_metrics.csv', index=False)
print("\n✓ Metrics saved to data/processed/plots/model_metrics.csv")

# ============================================================================
# 4. TIME SERIES PLOT - ALL MODELS (First 200 Days)
# ============================================================================
print("\n4. Creating time series plots...")

# Get first 200 samples
n_samples = min(200, len(test_df))
plot_df = test_df.head(n_samples).copy()
plot_df['index'] = range(len(plot_df))

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(plot_df['index'], plot_df['y_true'], 'k-', linewidth=2, label='Actual', alpha=0.9)
ax.plot(plot_df['index'], plot_df['lgbm_pred'], 'b-', linewidth=1.5, label='LightGBM', alpha=0.7)
ax.plot(plot_df['index'], plot_df['gru_pred'], 'g-', linewidth=1.5, label='GRU', alpha=0.7)
ax.plot(plot_df['index'], plot_df['hybrid_pred'], 'r-', linewidth=2, label='Hybrid (Ridge)', alpha=0.8)

ax.set_xlabel('Time Steps')
ax.set_ylabel('Temperature (°C)')
ax.set_title(f'Model Predictions Comparison (First {n_samples} Samples)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/plots/time_series_all_models.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Time series plot saved")

# ============================================================================
# 5. ZOOMED TIME SERIES PLOT (First 50 Days)
# ============================================================================
zoom_n = min(50, len(test_df))
zoom_df = test_df.head(zoom_n).copy()
zoom_df['index'] = range(len(zoom_df))

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(zoom_df['index'], zoom_df['y_true'], 'k-', linewidth=2, label='Actual', alpha=0.9)
ax.plot(zoom_df['index'], zoom_df['lgbm_pred'], 'b-', linewidth=1.5, label='LightGBM', alpha=0.7)
ax.plot(zoom_df['index'], zoom_df['gru_pred'], 'g-', linewidth=1.5, label='GRU', alpha=0.7)
ax.plot(zoom_df['index'], zoom_df['hybrid_pred'], 'r-', linewidth=2, label='Hybrid (Ridge)', alpha=0.8)

ax.set_xlabel('Time Steps')
ax.set_ylabel('Temperature (°C)')
ax.set_title(f'Zoomed View - First {zoom_n} Samples')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/plots/time_series_zoomed.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Zoomed time series plot saved")

# ============================================================================
# 6. PREDICTION ERRORS OVER TIME
# ============================================================================
# Calculate errors
test_df['lgbm_error'] = test_df['y_true'] - test_df['lgbm_pred']
test_df['gru_error'] = test_df['y_true'] - test_df['gru_pred']
test_df['hybrid_error'] = test_df['y_true'] - test_df['hybrid_pred']

error_df = test_df.head(200).copy()
error_df['index'] = range(len(error_df))

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(error_df['index'], error_df['lgbm_error'], 'b-', linewidth=1, label='LightGBM Error', alpha=0.7)
ax.plot(error_df['index'], error_df['gru_error'], 'g-', linewidth=1, label='GRU Error', alpha=0.7)
ax.plot(error_df['index'], error_df['hybrid_error'], 'r-', linewidth=1.5, label='Hybrid Error', alpha=0.8)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

ax.set_xlabel('Time Steps')
ax.set_ylabel('Error (°C)')
ax.set_title('Prediction Errors Over Time (First 200 Samples)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/plots/errors_over_time.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Error plot saved")

# ============================================================================
# 7. ERROR DISTRIBUTION (HISTOGRAM)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(test_df['lgbm_error'], bins=50, alpha=0.5, label=f'LightGBM', color='blue', density=True)
ax.hist(test_df['gru_error'], bins=50, alpha=0.5, label=f'GRU', color='green', density=True)
ax.hist(test_df['hybrid_error'], bins=50, alpha=0.5, label=f'Hybrid', color='red', density=True)
ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

ax.set_xlabel('Error (°C)')
ax.set_ylabel('Density')
ax.set_title('Error Distribution Comparison')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/plots/error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Error distribution saved")

# ============================================================================
# 8. SCATTER PLOTS - PREDICTED VS ACTUAL
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models_plot = [
    ('lgbm_pred', 'LightGBM', 'blue'),
    ('gru_pred', 'GRU', 'green'),
    ('hybrid_pred', 'Hybrid (Ridge)', 'red')
]

for idx, (model, name, color) in enumerate(models_plot):
    ax = axes[idx]
    
    ax.scatter(test_df['y_true'], test_df[model], alpha=0.5, s=10, c=color, label=f'{name} Predictions')
    
    min_val = min(test_df['y_true'].min(), test_df[model].min())
    max_val = max(test_df['y_true'].max(), test_df[model].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='Perfect Prediction')
    
    rmse = np.sqrt(mean_squared_error(test_df['y_true'], test_df[model]))
    r2 = r2_score(test_df['y_true'], test_df[model])
    ax.text(0.05, 0.95, f'RMSE: {rmse:.3f}°C\nR²: {r2:.3f}', 
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Actual Temperature (°C)')
    ax.set_ylabel(f'{name} Prediction (°C)')
    ax.set_title(f'{name}: Predicted vs Actual')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('data/processed/plots/scatter_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Scatter plots saved")

# ============================================================================
# 9. BOX PLOT OF ABSOLUTE ERRORS
# ============================================================================
error_df_abs = pd.DataFrame({
    'LightGBM': np.abs(test_df['lgbm_error']),
    'GRU': np.abs(test_df['gru_error']),
    'Hybrid': np.abs(test_df['hybrid_error'])
})

fig, ax = plt.subplots(figsize=(10, 6))
error_df_abs.boxplot(ax=ax, grid=True, patch_artist=True)
ax.set_ylabel('Absolute Error (°C)')
ax.set_title('Absolute Error Distribution by Model')
ax.grid(True, alpha=0.3)

means = error_df_abs.mean()
means.plot(style='ro', ax=ax, label='Mean')
ax.legend()

plt.tight_layout()
plt.savefig('data/processed/plots/boxplot_errors.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Box plot saved")

# ============================================================================
# 10. PERFORMANCE BY TEMPERATURE RANGE
# ============================================================================
test_df['temp_range'] = pd.cut(test_df['y_true'], 
                               bins=[0, 15, 25, 35, 50],
                               labels=['Cold (<15°C)', 'Mild (15-25°C)', 
                                      'Warm (25-35°C)', 'Hot (>35°C)'])

range_metrics = []
for temp_range in test_df['temp_range'].unique():
    if pd.notna(temp_range):
        subset = test_df[test_df['temp_range'] == temp_range]
        
        for model_name, model_col in [('LightGBM', 'lgbm_pred'), 
                                       ('GRU', 'gru_pred'), 
                                       ('Hybrid', 'hybrid_pred')]:
            rmse = np.sqrt(mean_squared_error(subset['y_true'], subset[model_col]))
            range_metrics.append({
                'Temperature Range': temp_range,
                'Model': model_name,
                'RMSE (°C)': rmse,
                'Samples': len(subset)
            })

range_metrics_df = pd.DataFrame(range_metrics)
pivot_df = range_metrics_df.pivot(index='Temperature Range', columns='Model', values='RMSE (°C)')

print("\n" + "="*80)
print("RMSE BY TEMPERATURE RANGE (°C)")
print("="*80)
print(pivot_df.to_string())

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
pivot_df.plot(kind='bar', ax=ax)
ax.set_xlabel('Temperature Range')
ax.set_ylabel('RMSE (°C)')
ax.set_title('Model Performance by Temperature Range')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('data/processed/plots/performance_by_temp_range.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Performance by temperature range saved")

# ============================================================================
# 11. IMPROVEMENT PERCENTAGE COMPARISON
# ============================================================================
lgbm_rmse = np.sqrt(mean_squared_error(test_df['y_true'], test_df['lgbm_pred']))
gru_rmse = np.sqrt(mean_squared_error(test_df['y_true'], test_df['gru_pred']))
hybrid_rmse = np.sqrt(mean_squared_error(test_df['y_true'], test_df['hybrid_pred']))

improvements = {
    'Hybrid vs LightGBM': ((lgbm_rmse - hybrid_rmse) / lgbm_rmse) * 100,
    'Hybrid vs GRU': ((gru_rmse - hybrid_rmse) / gru_rmse) * 100,
    'LightGBM vs GRU': ((gru_rmse - lgbm_rmse) / gru_rmse) * 100
}

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(improvements.keys(), improvements.values(), 
              color=['green', 'red', 'blue'])

for bar, value in zip(bars, improvements.values()):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.2f}%', ha='center', va='bottom')

ax.set_ylabel('Improvement (%)')
ax.set_title('Model Improvement Comparison')
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig('data/processed/plots/improvement_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Improvement comparison saved")

# ============================================================================
# 12. SAMPLE PREDICTIONS TABLE
# ============================================================================
sample_df = test_df.head(20)[['Date', 'City', 'y_true', 'lgbm_pred', 'gru_pred', 'hybrid_pred']].copy()
sample_df['Date'] = sample_df['Date'].dt.strftime('%Y-%m-%d')
sample_df['lgbm_error'] = np.abs(sample_df['y_true'] - sample_df['lgbm_pred'])
sample_df['gru_error'] = np.abs(sample_df['y_true'] - sample_df['gru_pred'])
sample_df['hybrid_error'] = np.abs(sample_df['y_true'] - sample_df['hybrid_pred'])

print("\n" + "="*80)
print("SAMPLE PREDICTIONS (First 20 Test Samples)")
print("="*80)
print(sample_df[['Date', 'City', 'y_true', 'lgbm_pred', 'gru_pred', 'hybrid_pred']].round(2).to_string(index=False))

# Save sample predictions
sample_df.to_csv('data/processed/plots/sample_predictions.csv', index=False)
print("\n✓ Sample predictions saved to data/processed/plots/sample_predictions.csv")

# ============================================================================
# 13. FINAL SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY REPORT")
print("="*80)

print(f"\nDataset Information:")
print(f"  Total Test Samples: {len(test_df)}")
print(f"  Temperature Range: [{test_df['y_true'].min():.2f}°C, {test_df['y_true'].max():.2f}°C]")
print(f"  Average Temperature: {test_df['y_true'].mean():.2f}°C")

print(f"\nModel Performance Summary:")
print(f"  LightGBM RMSE: {lgbm_rmse:.4f}°C")
print(f"  GRU RMSE: {gru_rmse:.4f}°C")
print(f"  Hybrid (Ridge) RMSE: {hybrid_rmse:.4f}°C")

print(f"\nImprovements:")
print(f"  Hybrid vs LightGBM: {improvements['Hybrid vs LightGBM']:.2f}% improvement")
print(f"  Hybrid vs GRU: {improvements['Hybrid vs GRU']:.2f}% improvement")

best_model = 'Hybrid (Ridge)' if hybrid_rmse < lgbm_rmse and hybrid_rmse < gru_rmse else \
             'LightGBM' if lgbm_rmse < gru_rmse else 'GRU'
print(f"\nBest Model: {best_model}")

# Create summary dataframe
summary = pd.DataFrame({
    'Metric': ['RMSE (°C)', 'MAE (°C)', 'R² Score', 'Best Temperature Range'],
    'LightGBM': [f"{lgbm_rmse:.4f}", 
                 f"{mean_absolute_error(test_df['y_true'], test_df['lgbm_pred']):.4f}",
                 f"{r2_score(test_df['y_true'], test_df['lgbm_pred']):.4f}",
                 f"{pivot_df['LightGBM'].idxmin()}"],
    'GRU': [f"{gru_rmse:.4f}",
            f"{mean_absolute_error(test_df['y_true'], test_df['gru_pred']):.4f}",
            f"{r2_score(test_df['y_true'], test_df['gru_pred']):.4f}",
            f"{pivot_df['GRU'].idxmin()}"],
    'Hybrid (Ridge)': [f"{hybrid_rmse:.4f}",
                       f"{mean_absolute_error(test_df['y_true'], test_df['hybrid_pred']):.4f}",
                       f"{r2_score(test_df['y_true'], test_df['hybrid_pred']):.4f}",
                       f"{pivot_df['Hybrid'].idxmin()}"]
})

print("\n" + "="*80)
print("DETAILED METRICS SUMMARY")
print("="*80)
print(summary.to_string(index=False))

# Save summary
summary.to_csv('data/processed/plots/summary_report.csv', index=False)
print("\n✓ Summary report saved to data/processed/plots/summary_report.csv")

# ============================================================================
# 14. SAVE FINAL PREDICTIONS
# ============================================================================
final_output = test_df[['Date', 'City', 'y_true', 'lgbm_pred', 'gru_pred', 'hybrid_pred']].copy()
final_output['Date'] = final_output['Date'].dt.strftime('%Y-%m-%d')
final_output.to_csv('data/processed/final_hybrid_predictions.csv', index=False)

print("\n✓ Final hybrid predictions saved to data/processed/final_hybrid_predictions.csv")

# ============================================================================
# 15. COMPLETE
# ============================================================================
print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print("\nAll visualizations have been saved to:")
print("  - data/processed/plots/time_series_all_models.png")
print("  - data/processed/plots/time_series_zoomed.png")
print("  - data/processed/plots/errors_over_time.png")
print("  - data/processed/plots/error_distribution.png")
print("  - data/processed/plots/scatter_plots.png")
print("  - data/processed/plots/boxplot_errors.png")
print("  - data/processed/plots/performance_by_temp_range.png")
print("  - data/processed/plots/improvement_comparison.png")
print("\nMetrics saved to:")
print("  - data/processed/plots/model_metrics.csv")
print("  - data/processed/plots/summary_report.csv")
print("  - data/processed/plots/sample_predictions.csv")
print("  - data/processed/final_hybrid_predictions.csv")