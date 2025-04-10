import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def evaluate_model(model_path, test_data_path, model_type='rf'):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path (str): Path to saved model
        test_data_path (str): Path to test data CSV
        model_type (str): Type of model ('rf' for Random Forest, 'nn' for Neural Network)
        
    Returns:
        dict: Dictionary containing evaluation metrics and plots
    """
    # Load model and test data
    if model_type == 'rf':
        model = joblib.load(model_path)
    else:  # Neural Network
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    
    test_df = pd.read_csv(test_data_path)
    
    # Prepare data
    X_test = test_df[['temperature', 'precipitation', 'co2', 'latitude', 'longitude', 'year', 'month']]
    y_test = test_df['temp_anomaly']
    
    # Load scaler
    scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.joblib')
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    if model_type == 'nn':
        y_pred = y_pred.flatten()
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'explained_variance': explained_variance_score(y_test, y_pred)
    }
    
    # Create plots
    plot_dir = os.path.join('reports', 'figures', datetime.now().strftime('%Y%m%d'))
    os.makedirs(plot_dir, exist_ok=True)
    
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    actual_vs_pred_path = os.path.join(plot_dir, 'actual_vs_predicted.png')
    plt.savefig(actual_vs_pred_path)
    plt.close()
    
    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    residual_path = os.path.join(plot_dir, 'residual_plot.png')
    plt.savefig(residual_path)
    plt.close()
    
    # Feature importance (for Random Forest)
    if model_type == 'rf' and hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        feature_imp_path = os.path.join(plot_dir, 'feature_importance.png')
        plt.savefig(feature_imp_path)
        plt.close()
        
        metrics['feature_importance'] = feature_importance.to_dict()
    
    return {
        'metrics': metrics,
        'plots': {
            'actual_vs_predicted': actual_vs_pred_path,
            'residual_plot': residual_path,
            'feature_importance': feature_imp_path if model_type == 'rf' else None
        }
    }

def generate_evaluation_report(model_type='rf'):
    """
    Generate a comprehensive evaluation report for a model.
    
    Args:
        model_type (str): Type of model ('rf' or 'nn')
    """
    model_path = f'models/{"random_forest" if model_type == "rf" else "tensorflow"}/{"rf_model.joblib" if model_type == "rf" else "nn_model"}'
    test_data_path = 'data/processed/climate_data.csv'  # Using full dataset with train_test_split in evaluation
    
    evaluation = evaluate_model(model_path, test_data_path, model_type)
    
    # Generate markdown report
    report_dir = os.path.join('reports', 'evaluations', datetime.now().strftime('%Y%m%d'))
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = os.path.join(report_dir, f'evaluation_report_{model_type}.md')
    
    with open(report_path, 'w') as f:
        f.write(f"# Model Evaluation Report ({'Random Forest' if model_type == 'rf' else 'Neural Network'})\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Evaluation Metrics\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for name, value in evaluation['metrics'].items():
            if name != 'feature_importance':
                f.write(f"| {name.upper()} | {value:.4f} |\n")
        
        if model_type == 'rf':
            f.write("\n## Feature Importance\n")
            f.write("| Feature | Importance |\n")
            f.write("|---------|------------|\n")
            for _, row in pd.DataFrame(evaluation['metrics']['feature_importance']).iterrows():
                f.write(f"| {row['feature']} | {row['importance']:.4f} |\n")
        
        f.write("\n## Visualization\n")
        f.write(f"![Actual vs Predicted]({os.path.relpath(evaluation['plots']['actual_vs_predicted'], report_dir)})\n\n")
        f.write(f"![Residual Plot]({os.path.relpath(evaluation['plots']['residual_plot'], report_dir)})\n")
        
        if model_type == 'rf':
            f.write(f"\n![Feature Importance]({os.path.relpath(evaluation['plots']['feature_importance'], report_dir)})\n")
    
    print(f"Evaluation report generated at: {report_path}")
    return report_path