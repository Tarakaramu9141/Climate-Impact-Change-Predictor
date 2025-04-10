import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_neural_network(data_path='data/processed/climate_data.csv'):
    """
    Train a Neural Network model on climate data.
    
    Args:
        data_path (str): Path to processed climate data
        
    Returns:
        tf.keras.Model: Trained model
    """
    try:
        # Load and prepare data
        logger.info("Loading data...")
        df = pd.read_csv(data_path)
        
        # Feature engineering
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Features and target
        X = df[['temperature', 'precipitation', 'co', 'latitude', 'longitude', 'year', 'month']]
        y = df['temperature']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build model
        logger.info("Building model...")
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train with early stopping
        logger.info("Training model...")
        early_stop = EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate
        y_pred = model.predict(X_test_scaled).flatten()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info("\nNeural Network Performance:")
        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"R2 Score: {r2:.4f}")
        
        # Save model and scaler
        os.makedirs('models/tensorflow', exist_ok=True)
        
        # Save model with .keras extension (recommended)
        model.save('models/tensorflow/nn_model.keras')  # Fixed: Added .keras extension
        joblib.dump(scaler, 'models/tensorflow/scaler.joblib')
        
        logger.info("Model saved successfully!")
        return model
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return None

if __name__ == '__main__':
    logger.info("Starting neural network training...")
    model = train_neural_network()
    if model:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")