import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

def train_random_forest(data_path='data/processed/climate_data.csv'):
    """
    Train a Random Forest model on climate data.
    
    Args:
        data_path (str): Path to processed climate data
        
    Returns:
        RandomForestRegressor: Trained model
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None
    
    # Feature engineering
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Features and target
    X = df[['temperature', 'precipitation', 'co', 'latitude', 'longitude', 'year', 'month']]
    y = df['temperature']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nRandom Forest Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save model
    os.makedirs('models/random_forest', exist_ok=True)
    joblib.dump(rf, 'models/random_forest/rf_model.joblib')
    joblib.dump(scaler, 'models/random_forest/scaler.joblib')
    
    return rf

if __name__ == '__main__':
    print("Training Random Forest model...")
    model = train_random_forest()
    if model is not None:
        print("Model training completed successfully!")