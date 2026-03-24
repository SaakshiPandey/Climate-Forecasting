"""
OPTIMAL RIDGE STACKING ENSEMBLE
Best performing method for LightGBM + GRU combination
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class OptimalRidgeEnsemble:
    """
    Optimized Ridge Stacking for combining LightGBM and GRU predictions
    """
    
    def __init__(self, val_df, test_df):
        """
        Initialize with validation and test dataframes
        Expected columns: Date, City, y_true, lgbm_pred, gru_pred
        """
        self.val_df = val_df.copy()
        self.test_df = test_df.copy()
        
        # Engineer features
        self._engineer_features()
        
        # Store results
        self.model = None
        self.scaler = None
        self.features = None
        self.val_rmse = None
        self.test_rmse = None
        
    def _engineer_features(self):
        """Create optimal features for stacking"""
        for df in [self.val_df, self.test_df]:
            # Base predictions
            df['lgbm_pred'] = df['lgbm_pred'].values
            df['gru_pred'] = df['gru_pred'].values
            
            # Interaction features
            df['pred_diff'] = df['lgbm_pred'] - df['gru_pred']
            df['pred_sum'] = df['lgbm_pred'] + df['gru_pred']
            df['pred_product'] = df['lgbm_pred'] * df['gru_pred']
            df['pred_mean'] = (df['lgbm_pred'] + df['gru_pred']) / 2
            
            # Polynomial features
            df['lgbm_squared'] = df['lgbm_pred'] ** 2
            df['gru_squared'] = df['gru_pred'] ** 2
            
            # Ratio features (with safety)
            df['pred_ratio'] = df['lgbm_pred'] / (df['gru_pred'] + 1e-8)
            
            # Confidence features (based on prediction agreement)
            df['pred_std'] = df[['lgbm_pred', 'gru_pred']].std(axis=1)
            df['pred_agreement'] = 1 / (df['pred_std'] + 0.01)
    
    def find_optimal_features(self):
        """
        Find the best combination of features through forward selection
        """
        print("Finding optimal feature combination...")
        
        # All candidate features
        candidate_features = [
            'lgbm_pred', 'gru_pred',           # Base predictions
            'pred_diff', 'pred_mean',           # Simple combinations
            'pred_product', 'pred_ratio',       # Interaction features
            'lgbm_squared', 'gru_squared',      # Polynomial features
            'pred_std', 'pred_agreement'        # Confidence features
        ]
        
        # Use only features that exist
        candidate_features = [f for f in candidate_features if f in self.val_df.columns]
        
        X_train = self.val_df[candidate_features].values
        y_train = self.val_df['y_true'].values
        
        # Forward feature selection with time series CV
        selected_features = []
        remaining_features = candidate_features.copy()
        best_cv_score = float('inf')
        
        # Try each feature individually first
        feature_scores = []
        for feat in remaining_features:
            X_temp = self.val_df[[feat]].values
            
            # Scale
            scaler_temp = StandardScaler()
            X_scaled = scaler_temp.fit_transform(X_temp)
            
            # Time series CV
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_fold_train, X_fold_val = X_scaled[train_idx], X_scaled[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                ridge = Ridge(alpha=0.01)
                ridge.fit(X_fold_train, y_fold_train)
                pred = ridge.predict(X_fold_val)
                rmse = np.sqrt(mean_squared_error(y_fold_val, pred))
                cv_scores.append(rmse)
            
            mean_score = np.mean(cv_scores)
            feature_scores.append((feat, mean_score))
        
        # Sort by performance
        feature_scores.sort(key=lambda x: x[1])
        
        print("\nTop 5 individual features:")
        for feat, score in feature_scores[:5]:
            print(f"  {feat}: {score:.4f}°C")
        
        # Select top 4 features (balance between performance and simplicity)
        selected_features = [f for f, _ in feature_scores[:4]]
        
        print(f"\nSelected features: {selected_features}")
        
        return selected_features
    
    def train(self, features=None, alpha=None):
        """
        Train the Ridge stacking model
        """
        print("\n" + "="*60)
        print("TRAINING RIDGE STACKING ENSEMBLE")
        print("="*60)
        
        # Select features
        if features is None:
            self.features = self.find_optimal_features()
        else:
            self.features = features
        
        # Prepare data
        X_train = self.val_df[self.features].values
        y_train = self.val_df['y_true'].values
        X_test = self.test_df[self.features].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Find optimal alpha with time series CV
        if alpha is None:
            print("\nFinding optimal regularization strength...")
            tscv = TimeSeriesSplit(n_splits=5)
            
            best_alpha = None
            best_cv_rmse = float('inf')
            cv_results = []
            
            for alpha_val in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X_train_scaled):
                    X_fold_train = X_train_scaled[train_idx]
                    y_fold_train = y_train[train_idx]
                    X_fold_val = X_train_scaled[val_idx]
                    y_fold_val = y_train[val_idx]
                    
                    ridge = Ridge(alpha=alpha_val)
                    ridge.fit(X_fold_train, y_fold_train)
                    pred = ridge.predict(X_fold_val)
                    rmse = np.sqrt(mean_squared_error(y_fold_val, pred))
                    cv_scores.append(rmse)
                
                mean_rmse = np.mean(cv_scores)
                cv_results.append((alpha_val, mean_rmse))
                
                if mean_rmse < best_cv_rmse:
                    best_cv_rmse = mean_rmse
                    best_alpha = alpha_val
            
            print(f"Best alpha: {best_alpha} (CV RMSE: {best_cv_rmse:.4f}°C)")
            alpha = best_alpha
        
        # Train final model
        self.model = Ridge(alpha=alpha)
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        self.val_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        self.test_rmse = np.sqrt(mean_squared_error(self.test_df['y_true'], test_pred))
        
        # Additional metrics
        val_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(self.test_df['y_true'], test_pred)
        val_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(self.test_df['y_true'], test_pred)
        
        # Display results
        print("\n" + "-"*60)
        print("MODEL RESULTS")
        print("-"*60)
        
        print("\nCoefficients:")
        for feat, coef in zip(self.features, self.model.coef_):
            print(f"  {feat:<20}: {coef:>8.4f}")
        print(f"  {'intercept':<20}: {self.model.intercept_:>8.4f}")
        
        print("\nMetrics:")
        print(f"  Validation RMSE: {self.val_rmse:.4f}°C")
        print(f"  Validation MAE:  {val_mae:.4f}°C")
        print(f"  Validation R²:   {val_r2:.4f}")
        print(f"\n  Test RMSE:       {self.test_rmse:.4f}°C")
        print(f"  Test MAE:        {test_mae:.4f}°C")
        print(f"  Test R²:         {test_r2:.4f}")
        
        # Compare with individual models
        lgbm_rmse = np.sqrt(mean_squared_error(self.test_df['y_true'], self.test_df['lgbm_pred']))
        gru_rmse = np.sqrt(mean_squared_error(self.test_df['y_true'], self.test_df['gru_pred']))
        
        print("\nComparison with individual models:")
        print(f"  LightGBM:        {lgbm_rmse:.4f}°C")
        print(f"  GRU:             {gru_rmse:.4f}°C")
        print(f"  Ridge Ensemble:  {self.test_rmse:.4f}°C")
        
        improvement_lgbm = (lgbm_rmse - self.test_rmse) / lgbm_rmse * 100
        improvement_gru = (gru_rmse - self.test_rmse) / gru_rmse * 100
        
        if self.test_rmse < lgbm_rmse:
            print(f"\n✓ Improvement over LightGBM: {improvement_lgbm:.2f}%")
        if self.test_rmse < gru_rmse:
            print(f"✓ Improvement over GRU: {improvement_gru:.2f}%")
        
        return test_pred
    
    def predict(self, new_lgbm, new_gru):
        """
        Make predictions on new data
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create dataframe with predictions
        df = pd.DataFrame({
            'lgbm_pred': new_lgbm,
            'gru_pred': new_gru
        })
        
        # Engineer features
        df['pred_diff'] = df['lgbm_pred'] - df['gru_pred']
        df['pred_sum'] = df['lgbm_pred'] + df['gru_pred']
        df['pred_product'] = df['lgbm_pred'] * df['gru_pred']
        df['pred_mean'] = (df['lgbm_pred'] + df['gru_pred']) / 2
        df['lgbm_squared'] = df['lgbm_pred'] ** 2
        df['gru_squared'] = df['gru_pred'] ** 2
        df['pred_ratio'] = df['lgbm_pred'] / (df['gru_pred'] + 1e-8)
        df['pred_std'] = df[['lgbm_pred', 'gru_pred']].std(axis=1)
        df['pred_agreement'] = 1 / (df['pred_std'] + 0.01)
        
        # Select features
        X = df[self.features].values
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def save_model(self, path='models/ridge_ensemble.pkl'):
        """
        Save the trained model
        """
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features,
            'val_rmse': self.val_rmse,
            'test_rmse': self.test_rmse
        }
        
        joblib.dump(model_data, path)
        print(f"\n✓ Model saved to {path}")
    
    def save_predictions(self, predictions, output_path='data/processed/best_hybrid_predictions.csv'):
        """
        Save predictions to CSV
        """
        output = self.test_df[['Date', 'City', 'y_true']].copy()
        output['lgbm_pred'] = self.test_df['lgbm_pred']
        output['gru_pred'] = self.test_df['gru_pred']
        output['hybrid_pred'] = predictions
        
        # Add error column
        output['error'] = output['y_true'] - output['hybrid_pred']
        
        output.to_csv(output_path, index=False)
        print(f"\n✓ Predictions saved to {output_path}")
        
        return output


# ==================== MAIN EXECUTION ====================
def main():
    """
    Run the optimal ridge stacking ensemble
    """
    print("="*70)
    print("OPTIMAL RIDGE STACKING ENSEMBLE")
    print("LightGBM + GRU Temperature Forecasting")
    print("="*70)
    
    # Load data
    print("\n1. Loading prediction files...")
    
    # Validation data
    val_df = pd.read_csv('data/processed/lgbm_val_preds.csv')
    val_df['Date'] = pd.to_datetime(val_df['Date'])
    
    gru_val = pd.read_csv('data/processed/gru_val_preds.csv')
    gru_val['Date'] = pd.to_datetime(gru_val['Date'])
    val_df = val_df.merge(gru_val[['Date', 'City', 'gru_pred']], on=['Date', 'City'])
    
    # Test data
    test_df = pd.read_csv('data/processed/lgbm_test_preds.csv')
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    
    gru_test = pd.read_csv('data/processed/gru_test_preds.csv')
    gru_test['Date'] = pd.to_datetime(gru_test['Date'])
    test_df = test_df.merge(gru_test[['Date', 'City', 'gru_pred']], on=['Date', 'City'])
    
    print(f"✓ Validation samples: {len(val_df)}")
    print(f"✓ Test samples: {len(test_df)}")
    
    # Train ensemble
    print("\n2. Training Ridge Stacking Ensemble...")
    ensemble = OptimalRidgeEnsemble(val_df, test_df)
    predictions = ensemble.train()
    
    # Save results
    print("\n3. Saving results...")
    ensemble.save_predictions(predictions)
    ensemble.save_model()
    
    # Display sample predictions
    print("\n4. Sample predictions (first 10 test samples):")
    sample = test_df.head(10)[['Date', 'City', 'y_true', 'lgbm_pred', 'gru_pred']].copy()
    sample['hybrid_pred'] = predictions[:10]
    sample['error'] = sample['y_true'] - sample['hybrid_pred']
    print(sample.to_string(index=False))
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"✅ Best Model: Ridge Stacking Ensemble")
    print(f"✅ Test RMSE: {ensemble.test_rmse:.4f}°C")
    print(f"✅ Model saved to: models/ridge_ensemble.pkl")
    print(f"✅ Predictions saved to: data/processed/best_hybrid_predictions.csv")
    print("="*70)
    
    return ensemble, predictions


if __name__ == "__main__":
    ensemble, predictions = main()