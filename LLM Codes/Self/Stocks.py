# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
import tensorflow as tf
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Module 1: Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    """
    Load and preprocess stock data from CSV

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    df (DataFrame): Preprocessed dataframe
    """
    print("🔹 Loading data into DataFrame...")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    print(df.info())
    print(df.describe())

    # Print date range in the dataset
    print(f"Dataset date range: {df.index.min()} to {df.index.max()}")

    # Check for missing values and handle them
    print("🔹 Checking for missing values...")
    missing_values = df.isnull().sum()
    print(f"Missing values per column:\n{missing_values}")

    if df.isnull().any().any():
        plt.figure(figsize=(12,6))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title("Missing Data Heatmap")
        plt.tight_layout()
        plt.show()

        # Use interpolate(method='time') for missing values
        print("🔹 Filling missing values using time interpolation...")
        df['Close'] = df['Close'].interpolate(method='time')

    return df

# Module 2: Technical Indicators
def calculate_rsi(prices, window=14):
    """
    Calculate the Relative Strength Index (RSI) technical indicator

    Parameters:
    prices (Series): Series of prices
    window (int): RSI window length

    Returns:
    Series: RSI values
    """
    # Calculate price differences
    delta = prices.diff()

    # Create gain (positive) and loss (negative) Series
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gain and loss over specified window
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe

    Parameters:
    df (DataFrame): Input dataframe with stock data

    Returns:
    df (DataFrame): Dataframe with technical indicators
    """
    print("🔹 Adding technical indicators...")
    # Add moving averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # Add volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()

    # Add momentum indicators
    df['ROC'] = df['Close'].pct_change(periods=5) * 100  # Rate of change
    df['RSI'] = calculate_rsi(df['Close'], window=14)

    # Add volume indicators
    if 'Volume' in df.columns:
        df['Volume_Change'] = df['Volume'].pct_change() * 100
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()

    # Drop rows with NaN values after feature engineering
    df.dropna(inplace=True)

    return df

# Module 3: Data Preparation for LSTM
def prepare_data_for_lstm(df):
    """
    Prepare data for LSTM model

    Parameters:
    df (DataFrame): Input dataframe with features

    Returns:
    tuple: features, scaled_data, scaler, close_scaler
    """
    print("🔹 Preparing data for LSTM...")

    # Select features
    features = ['Close', 'MA_7', 'MA_20', 'MA_50', 'Volatility', 'ROC']

    # Add volume features if available
    if 'Volume' in df.columns:
        features.extend(['Volume_Change', 'Volume_MA_5'])

    # Display selected features
    print(f"Selected features: {features}")
    data = df[features].values

    # Scale all features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Close price scaler - will be used for inverse transforming predictions
    close_scaler = MinMaxScaler()
    close_scaler.fit_transform(df[['Close']])

    return features, scaled_data, scaler, close_scaler

def create_sequences(data, lookback=60):
    """
    Create sequences of lookback days for LSTM input

    Parameters:
    data (numpy.array): Scaled input data
    lookback (int): Number of previous time steps to use as input for predicting the next step

    Returns:
    X (numpy.array): Input sequences
    y (numpy.array): Target values (Close price)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])  # Target is the Close price (index 0)
    return np.array(X), np.array(y)

def split_data_by_date(df, X, y, lookback, train_start, train_end, test_start, test_end):
    """
    Split data based on date ranges

    Parameters:
    df (DataFrame): Input dataframe
    X (numpy.array): Sequence data
    y (numpy.array): Target values
    lookback (int): Sequence length
    train_start, train_end, test_start, test_end (Timestamp): Date ranges

    Returns:
    tuple: X_train, y_train, X_test, y_test, train_df, test_df
    """
    print("🔹 Splitting data into train/test sets...")

    # Filter dataframe for train and test periods
    train_df = df[(df.index >= train_start) & (df.index <= train_end)]
    test_df = df[(df.index >= test_start) & (df.index <= test_end)]

    # Verify the split
    print(f"Training data: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} records)")
    print(f"Testing data: {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} records)")

    # Find indices corresponding to the date ranges
    train_indices = df[(df.index >= train_start) & (df.index <= train_end)].index
    test_indices = df[(df.index >= test_start) & (df.index <= test_end)].index

    # Get positions in the original dataframe
    train_positions = [df.index.get_loc(idx) for idx in train_indices]
    test_positions = [df.index.get_loc(idx) for idx in test_indices]

    # Adjust for lookback
    train_X_indices = [i for i in train_positions if i >= lookback]
    test_X_indices = [i for i in test_positions if i >= lookback]

    # Calculate offsets
    offset = lookback
    train_X_positions = [i - offset for i in train_X_indices]
    test_X_positions = [i - offset for i in test_X_indices]

    # Create train/test sets
    X_train = X[train_X_positions]
    y_train = y[train_X_positions]
    X_test = X[test_X_positions]
    y_test = y[test_X_positions]

    print(f"✅ Train Samples: {X_train.shape}, Test Samples: {X_test.shape}")

    return X_train, y_train, X_test, y_test, train_df, test_df

# Module 4: Model Building (with ReLU and GELU activations)
def build_model_relu(features_count, lookback, lr=0.001):
    """
    Build LSTM model with ReLU activation

    Parameters:
    features_count (int): Number of features in the input data
    lookback (int): Sequence length
    lr (float): Learning rate for the optimizer

    Returns:
    model: Compiled Keras LSTM model
    """
    model = Sequential([
        # First LSTM layer with return sequences for stacking
        LSTM(128, return_sequences=True,
             input_shape=(lookback, features_count),
             recurrent_dropout=0.1),
        Dropout(0.3),

        # Second LSTM layer
        LSTM(128, return_sequences=False, recurrent_dropout=0.1),
        Dropout(0.3),

        # Dense hidden layer with ReLU activation
        Dense(64, activation='relu'),
        Dropout(0.2),

        # Output layer (no activation for regression task)
        Dense(1)
    ])

    # Compile the model with Adam optimizer and MSE loss
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')

    # Print model summary
    model.summary()

    return model

def build_model_gelu(features_count, lookback, lr=0.001):
    """
    Build LSTM model with GELU activation

    Parameters:
    features_count (int): Number of features in the input data
    lookback (int): Sequence length
    lr (float): Learning rate for the optimizer

    Returns:
    model: Compiled Keras LSTM model
    """
    def gelu(x):
        # Implementation of the Gaussian Error Linear Unit activation function
        return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

    model = Sequential([
        # First LSTM layer with return sequences for stacking
        LSTM(128, return_sequences=True,
             input_shape=(lookback, features_count),
             recurrent_dropout=0.1),
        Dropout(0.3),

        # Second LSTM layer
        LSTM(128, return_sequences=False, recurrent_dropout=0.1),
        Dropout(0.3),

        # Dense hidden layer with GELU activation
        Dense(64, activation=gelu),
        Dropout(0.2),

        # Output layer (no activation for regression task)
        Dense(1)
    ])

    # Compile the model with Adam optimizer and MSE loss
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')

    # Print model summary
    model.summary()

    return model

# Module 5: Model Training and Evaluation
def get_callbacks():
    """
    Create callbacks for training

    Returns:
    list: Keras callbacks
    """
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,              # More patient
        restore_best_weights=True,
        verbose=1
    )

    # Learning rate reduction when performance plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,              # Halve the learning rate when plateauing
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    return [early_stop, reduce_lr]

def train_model(model, X_train, y_train, X_test, y_test, callbacks, model_name='lstm_model'):
    """
    Train LSTM model

    Parameters:
    model: Keras model
    X_train, y_train, X_test, y_test: Training and testing data
    callbacks: Keras callbacks
    model_name (str): Name for model saving

    Returns:
    history: Keras training history object
    """
    print(f"🔹 Training {model_name}...")

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Save model in the recommended Keras format (.keras)
    model.save(f'{model_name}.keras')
    print(f"✅ {model_name} saved successfully!")

    return history

def plot_training_history(history, title="Training History"):
    """
    Plot training and validation loss

    Parameters:
    history: Keras training history object
    title (str): Plot title
    """
    plt.figure(figsize=(14,6))

    # Training vs Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Training and Validation Loss')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Zoomed view of final epochs
    plt.subplot(1, 2, 2)
    window = min(30, len(history.history['val_loss']))  # Last 30 epochs or all if less
    offset = max(0, len(history.history['val_loss']) - window)

    plt.plot(range(offset, len(history.history['val_loss'])),
             history.history['val_loss'][offset:], label='Validation Loss')
    plt.title('Final Epochs (Zoomed)')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def predict_and_evaluate(model, X, y, scaler, features, df_subset, model_name='Model'):
    """
    Make predictions and evaluate model performance

    Parameters:
    model: Trained Keras model
    X (numpy.array): Input sequences
    y (numpy.array): True values (scaled)
    scaler: Fitted MinMaxScaler
    features (list): List of feature names
    df_subset (DataFrame): DataFrame containing the actual close prices for the time period
    model_name (str): Name of the model for printing

    Returns:
    actual (numpy.array): Actual stock prices
    pred (numpy.array): Predicted stock prices
    metrics (dict): Evaluation metrics
    results_df (DataFrame): DataFrame with predictions and errors
    """
    print(f"\n📊 {model_name} Evaluation:")

    # Make predictions
    pred_scaled = model.predict(X)

    # Prepare for inverse transformation
    # Create a dummy array matching the original feature dimensions
    dummy = np.zeros((len(pred_scaled), len(features)))
    # Put predictions in the first column (Close price)
    dummy[:, 0] = pred_scaled.flatten()

    # Inverse transform
    pred = scaler.inverse_transform(dummy)[:, 0]  # Get only the Close price column

    # Original values - also need to be extracted from the full feature set
    dummy_y = np.zeros((len(y), len(features)))
    dummy_y[:, 0] = y  # Put actual values in Close price position
    actual = scaler.inverse_transform(dummy_y)[:, 0]  # Get only the Close price

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    r2 = r2_score(actual, pred)

    # Calculate accuracy percentage
    accuracy_percentage = 100 - mape

    # Print metrics
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Accuracy Percentage: {accuracy_percentage:.2f}%")
    print(f"R² Score: {r2:.4f}")

    # Create a DataFrame with predictions for further analysis
    results_df = pd.DataFrame(
        data={
            'Actual': actual,
            'Predicted': pred,
            'Error': actual - pred,
            'Abs_Error': np.abs(actual - pred),
            'Pct_Error': np.abs((actual - pred) / actual) * 100
        },
        index=df_subset.index[-len(actual):]
    )

    # Calculate additional statistics
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'accuracy_percentage': accuracy_percentage,
        'r2': r2,
        'max_error': results_df['Abs_Error'].max(),
        'min_error': results_df['Abs_Error'].min(),
        'mean_error': results_df['Error'].mean(),
        'median_error': results_df['Error'].median(),
        'std_error': results_df['Error'].std()
    }

    # Print additional insights
    print(f"Maximum Absolute Error: {metrics['max_error']:.2f}")
    print(f"Minimum Absolute Error: {metrics['min_error']:.2f}")
    print(f"Mean Error (bias): {metrics['mean_error']:.2f}")
    print(f"Error Standard Deviation: {metrics['std_error']:.2f}")

    # Save results
    results_df.to_csv(f'{model_name.lower().replace(" ", "_")}_predictions.csv')

    return actual, pred, metrics, results_df

# Module 6: Visualizations
def plot_predictions_vs_actual(actual, predictions, dates, metrics, title="Stock Price Prediction"):
    """
    Plot actual vs predicted stock prices

    Parameters:
    actual (numpy.array): Actual stock prices
    predictions (numpy.array): Predicted stock prices
    dates (DatetimeIndex): Dates corresponding to the predictions
    metrics (dict): Evaluation metrics
    title (str): Plot title
    """
    # Predictions vs Actual Plot
    plt.figure(figsize=(16, 10))

    # Main prediction plot
    plt.subplot(2, 1, 1)
    plt.plot(dates, actual, label='Actual', linewidth=2)
    plt.plot(dates, predictions, label='Predicted', linewidth=2, linestyle='--')
    plt.title(f'{title}\nRMSE: {metrics["rmse"]:.2f}, MAPE: {metrics["mape"]:.2f}%',
              fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Error plot
    plt.subplot(2, 1, 2)
    error = actual - predictions
    plt.bar(dates, error, alpha=0.7, color='red', label='Error (Actual - Predicted)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Prediction Error', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Error (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Scatter plot: Predicted vs Actual
    plt.figure(figsize=(10, 8))
    plt.scatter(actual, predictions, alpha=0.5)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')  # Perfect prediction line
    plt.title(f'Scatter Plot: {title}', fontsize=14)
    plt.xlabel('Actual Price (USD)', fontsize=12)
    plt.ylabel('Predicted Price (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add R² annotation
    plt.annotate(f'R² = {metrics["r2"]:.4f}',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, bbox=dict(boxstyle='round', alpha=0.1))

    plt.tight_layout()
    plt.show()

    # Distribution of prediction errors
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(error, kde=True)
    plt.title(f'Distribution of Prediction Errors - {title}', fontsize=14)
    plt.xlabel('Error (USD)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    sns.boxplot(y=error)
    plt.title('Boxplot of Prediction Errors', fontsize=14)
    plt.ylabel('Error (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def analyze_monthly_performance(results_df, title="Monthly Performance"):
    """
    Analyze model performance on a monthly basis

    Parameters:
    results_df (DataFrame): DataFrame with actual and predicted values
    title (str): Plot title

    Returns:
    monthly_metrics (DataFrame): Monthly performance metrics
    """
    # Add month and year columns
    results_df['Year'] = results_df.index.year
    results_df['Month'] = results_df.index.month

    # Group by year and month
    monthly_metrics = results_df.groupby(['Year', 'Month']).agg({
        'Actual': 'mean',
        'Predicted': 'mean',
        'Abs_Error': 'mean',
        'Pct_Error': 'mean'
    }).reset_index()

    # Add month name
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    monthly_metrics['Month_Name'] = monthly_metrics['Month'].map(month_names)
    monthly_metrics['Period'] = monthly_metrics['Year'].astype(str) + '-' + monthly_metrics['Month_Name']

    # Plot monthly MAPE
    plt.figure(figsize=(16, 6))
    plt.bar(monthly_metrics['Period'], monthly_metrics['Pct_Error'], color='skyblue')
    plt.title(f'{title} - Monthly Mean Absolute Percentage Error (MAPE)', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('MAPE (%)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)

    # Add average line
    avg_mape = monthly_metrics['Pct_Error'].mean()
    plt.axhline(y=avg_mape, color='red', linestyle='--', label=f'Average: {avg_mape:.2f}%')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return monthly_metrics

def predict_future(model, scaler, last_sequence, days=30, features_count=6):
    """
    Predict future stock prices using the latest available data

    Parameters:
    model: Trained Keras model
    scaler: Fitted MinMaxScaler
    last_sequence: Last known sequence of data points
    days (int): Number of days to predict into the future
    features_count (int): Number of features in the model

    Returns:
    future_prices (numpy.array): Predicted future prices
    """
    # Copy the last sequence to avoid modifying the original
    sequence = last_sequence.copy()
    future_preds = []

    for _ in range(days):
        # Predict next value
        next_pred = model.predict(sequence)
        future_preds.append(next_pred[0, 0])

        # Update sequence for next prediction
        # Move window forward by shifting values and adding the new prediction
        new_input = np.zeros((1, 1, features_count))
        new_input[0, 0, 0] = next_pred  # Add prediction as Close price

        # Shift the sequence and add new value
        sequence = np.concatenate([
            sequence[:, 1:, :],  # Remove oldest entry
            new_input  # Add newest prediction
        ], axis=1)

    # Create dummy array for inverse transform
    dummy = np.zeros((len(future_preds), features_count))
    dummy[:, 0] = future_preds  # Put predictions in Close price position

    # Inverse transform to get actual price values
    future_prices = scaler.inverse_transform(dummy)[:, 0]

    return future_prices

def plot_future_predictions(actual, test_dates, future_predictions, future_dates, title="Future Prediction"):
    """
    Plot future predictions

    Parameters:
    actual (numpy.array): Actual stock prices
    test_dates (DatetimeIndex): Dates corresponding to actual prices
    future_predictions (numpy.array): Predicted future prices
    future_dates (DatetimeIndex): Dates corresponding to future predictions
    title (str): Plot title
    """
    plt.figure(figsize=(14, 6))
    # Plot recent history (last 60 days)
    plt.plot(test_dates[-60:], actual[-60:], label='Historical Close Price', color='blue')
    # Plot predictions
    plt.plot(future_dates, future_predictions, label='Future Prediction', color='red', linestyle='--')

    plt.title(f'{title} (Next {len(future_dates)} Days)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Module 7: Model Comparison
def compare_models(relu_metrics, gelu_metrics):
    """
    Compare performance metrics between ReLU and GELU models

    Parameters:
    relu_metrics (dict): Metrics from ReLU model
    gelu_metrics (dict): Metrics from GELU model
    """
    # Extract metrics
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'MAPE (%)', 'Accuracy (%)', 'R²',
                  'Max Error', 'Min Error', 'Mean Error', 'Median Error', 'Std Error'],
        'ReLU': [
            relu_metrics['rmse'],
            relu_metrics['mae'],
            relu_metrics['mape'],
            relu_metrics['accuracy_percentage'],
            relu_metrics['r2'],
            relu_metrics['max_error'],
            relu_metrics['min_error'],
            relu_metrics['mean_error'],
            relu_metrics['median_error'],
            relu_metrics['std_error']
        ],
        'GELU': [
            gelu_metrics['rmse'],
            gelu_metrics['mae'],
            gelu_metrics['mape'],
            gelu_metrics['accuracy_percentage'],
            gelu_metrics['r2'],
            gelu_metrics['max_error'],
            gelu_metrics['min_error'],
            gelu_metrics['mean_error'],
            gelu_metrics['median_error'],
            gelu_metrics['std_error']
        ]
    })

    # Calculate differences and percentage improvements
    metrics_df['Difference (GELU-ReLU)'] = metrics_df['GELU'] - metrics_df['ReLU']

    # For percentage improvement, handle differently based on metric
    # For error metrics (lower is better), negative percentage means improvement
    # For accuracy and R² (higher is better), positive percentage means improvement
    improvement = []

    for i, metric in enumerate(metrics_df['Metric']):
        if metric in ['Accuracy (%)', 'R²']:
            # Higher is better
            pct_change = ((metrics_df['GELU'][i] - metrics_df['ReLU'][i]) / metrics_df['ReLU'][i]) * 100
            improvement.append(f"{pct_change:+.2f}%")
        else:
            # Lower is better
            pct_change = ((metrics_df['GELU'][i] - metrics_df['ReLU'][i]) / metrics_df['ReLU'][i]) * 100
            improvement.append(f"{pct_change:+.2f}%")

    metrics_df['Improvement'] = improvement

    print("📊 Model Comparison: ReLU vs GELU")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Return comparison data
    return metrics_df

def plot_metric_comparison(relu_metrics, gelu_metrics):
    """
    Plot comparison of key metrics between ReLU and GELU models

    Parameters:
    relu_metrics (dict): Metrics from ReLU model
    gelu_metrics (dict): Metrics from GELU model
    """
    metrics = ['rmse', 'mae', 'mape', 'accuracy_percentage', 'r2']
    metric_names = ['RMSE', 'MAE', 'MAPE (%)', 'Accuracy (%)', 'R²']

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()

    # Plot metrics visualization
    relu_values = [relu_metrics[m] for m in metrics]
    gelu_values = [gelu_metrics[m] for m in metrics]

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        axs[i].bar(['ReLU', 'GELU'], [relu_metrics[metric], gelu_metrics[metric]], color=['blue', 'green'])
        axs[i].set_title(f'{name} Comparison', fontsize=14)
        axs[i].grid(True, alpha=0.3)

        # Add percentage difference
        diff_pct = ((gelu_metrics[metric] - relu_metrics[metric]) / relu_metrics[metric]) * 100

        # For RMSE, MAE, MAPE - lower is better, so negative percentage is improvement
        # For Accuracy, R² - higher is better, so positive percentage is improvement
        if metric in ['accuracy_percentage', 'r2']:
            color = 'green' if diff_pct > 0 else 'red'
            label = 'Better' if diff_pct > 0 else 'Worse'
        else:
            color = 'green' if diff_pct < 0 else 'red'
            label = 'Better' if diff_pct < 0 else 'Worse'

        axs[i].annotate(f"{diff_pct:+.2f}% ({label})",
                        xy=(1, gelu_metrics[metric]),
                        xytext=(5, 0),
                        textcoords='offset points',
                        ha='left',
                        va='center',
                        color=color,
                        fontweight='bold')

    # Remove extra subplot
    if len(axs) > len(metrics):
        fig.delaxes(axs[-1])

    plt.tight_layout()
    plt.suptitle('ReLU vs GELU Model Performance Metrics', fontsize=16, y=1.02)
    plt.show()

def plot_predictions_comparison(test_dates, actual, relu_pred, gelu_pred):
    """
    Plot comparison of predictions between ReLU and GELU models

    Parameters:
    test_dates (DatetimeIndex): Dates for testing period
    actual (numpy.array): Actual stock prices
    relu_pred (numpy.array): Predictions from ReLU model
    gelu_pred (numpy.array): Predictions from GELU model
    """
    plt.figure(figsize=(16, 8))
    plt.plot(test_dates, actual, label='Actual', linewidth=2)
    plt.plot(test_dates, relu_pred, label='ReLU Prediction', linewidth=1.5, linestyle='--')
    plt.plot(test_dates, gelu_pred, label='GELU Prediction', linewidth=1.5, linestyle=':')

    plt.title('ReLU vs GELU Model Predictions', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Error comparison
    plt.figure(figsize=(16, 8))

    relu_error = actual - relu_pred
    gelu_error = actual - gelu_pred

    plt.plot(test_dates, relu_error, label='ReLU Error', alpha=0.7)
    plt.plot(test_dates, gelu_error, label='GELU Error', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.title('Error Comparison: ReLU vs GELU', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Error (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Error distribution comparison
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(relu_error, kde=True, color='blue', label='ReLU')
    sns.histplot(gelu_error, kde=True, color='green', alpha=0.6, label='GELU')
    plt.title('Error Distribution Comparison', fontsize=14)
    plt.xlabel('Error (USD)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=[relu_error, gelu_error], palette=['blue', 'green'])
    plt.xticks([0, 1], ['ReLU', 'GELU'])
    plt.title('Error Boxplot Comparison', fontsize=14)
    plt.ylabel('Error (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def visualize_loss_comparison(relu_history, gelu_history):
    """
    Compare training history between ReLU and GELU models

    Parameters:
    relu_history: Training history from ReLU model
    gelu_history: Training history from GELU model
    """
    plt.figure(figsize=(16, 6))

    # Training Loss Comparison
    plt.subplot(1, 2, 1)
    plt.plot(relu_history.history['loss'], label='ReLU Training Loss')
    plt.plot(gelu_history.history['loss'], label='GELU Training Loss')
    plt.title('Training Loss Comparison', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Validation Loss Comparison
    plt.subplot(1, 2, 2)
    plt.plot(relu_history.history['val_loss'], label='ReLU Validation Loss')
    plt.plot(gelu_history.history['val_loss'], label='GELU Validation Loss')
    plt.title('Validation Loss Comparison', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Main execution function
def main():
    """
    Main execution function to run the stock price prediction models
    """
    # 1. Load and preprocess data
    df = load_and_preprocess_data("GOOG.csv")

    # 2. Visualize the initial data
    plt.figure(figsize=(14,6))
    plt.plot(df.index, df['Close'], label='Closing Price')
    plt.title('GOOG Stock Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Add technical indicators
    df = add_technical_indicators(df)

    # 4. Prepare data for LSTM
    features, scaled_data, scaler, close_scaler = prepare_data_for_lstm(df)

    # 5. Create sequences
    lookback = 60
    print(f"🔹 Creating LSTM sequences (lookback={lookback} days)...")
    X, y = create_sequences(scaled_data, lookback)

    # 6. Split data based on dates
    train_start = pd.Timestamp("2004-01-01")
    train_end = pd.Timestamp("2017-12-31")
    test_start = pd.Timestamp("2018-01-01")
    test_end = pd.Timestamp("2020-12-31")

    X_train, y_train, X_test, y_test, train_df, test_df = split_data_by_date(
        df, X, y, lookback, train_start, train_end, test_start, test_end
    )

    # 7. Build and train ReLU model
    print("🔹 Building LSTM model with ReLU activation...")
    relu_model = build_model_relu(len(features), lookback, lr=0.001)
    callbacks = get_callbacks()
    relu_history = train_model(
        relu_model, X_train, y_train, X_test, y_test,
        callbacks, model_name='goog_stock_lstm_relu'
    )

    # 8. Build and train GELU model
    print("🔹 Building LSTM model with GELU activation...")
    gelu_model = build_model_gelu(len(features), lookback, lr=0.001)
    gelu_history = train_model(
        gelu_model, X_train, y_train, X_test, y_test,
        callbacks, model_name='goog_stock_lstm_gelu'
    )

    # 9. Plot training history for both models
    plot_training_history(relu_history, "ReLU Model")
    plot_training_history(gelu_history, "GELU Model")

    # 10. Evaluate both models on test data
    actual_relu, predictions_relu, metrics_relu, results_df_relu = predict_and_evaluate(
        relu_model, X_test, y_test, scaler, features, test_df, model_name='ReLU Model'
    )

    actual_gelu, predictions_gelu, metrics_gelu, results_df_gelu = predict_and_evaluate(
        gelu_model, X_test, y_test, scaler, features, test_df, model_name='GELU Model'
    )

    # 11. Visualize predictions for both models
    test_dates = results_df_relu.index

    plot_predictions_vs_actual(
        actual_relu, predictions_relu, test_dates, metrics_relu,
        title="GOOG Stock Price - ReLU Model"
    )

    plot_predictions_vs_actual(
        actual_gelu, predictions_gelu, test_dates, metrics_gelu,
        title="GOOG Stock Price - GELU Model"
    )

    # 12. Monthly analysis for both models
    monthly_metrics_relu = analyze_monthly_performance(
        results_df_relu, title="ReLU Model"
    )

    monthly_metrics_gelu = analyze_monthly_performance(
        results_df_gelu, title="GELU Model"
    )

    # 13. Future predictions for both models
    future_days = 30

    # ReLU model future prediction
    print(f"🔹 Generating future predictions with ReLU model (next {future_days} days)...")
    last_known_sequence_relu = X_test[-1:].copy()
    future_predictions_relu = predict_future(
        relu_model, scaler, last_known_sequence_relu, future_days, len(features)
    )

    # GELU model future prediction
    print(f"🔹 Generating future predictions with GELU model (next {future_days} days)...")
    last_known_sequence_gelu = X_test[-1:].copy()
    future_predictions_gelu = predict_future(
        gelu_model, scaler, last_known_sequence_gelu, future_days, len(features)
    )

    # Calculate future dates starting from the last date in the test set
    last_date = test_dates[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    # Plot future predictions for both models
    plot_future_predictions(
        actual_relu, test_dates, future_predictions_relu, future_dates,
        title="GOOG Stock Price Future Prediction - ReLU Model"
    )

    plot_future_predictions(
        actual_gelu, test_dates, future_predictions_gelu, future_dates,
        title="GOOG Stock Price Future Prediction - GELU Model"
    )

    # 14. Model comparison
    metrics_comparison = compare_models(metrics_relu, metrics_gelu)
    metrics_comparison.to_csv('relu_vs_gelu_comparison.csv', index=False)

    # 15. Visual comparisons
    visualize_loss_comparison(relu_history, gelu_history)
    plot_metric_comparison(metrics_relu, metrics_gelu)
    plot_predictions_comparison(test_dates, actual_relu, predictions_relu, predictions_gelu)

    # 16. Compare future predictions
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates[-30:], actual_relu[-30:], label='Historical Close Price', color='blue')
    plt.plot(future_dates, future_predictions_relu, label='ReLU Future Prediction', color='green', linestyle='--')
    plt.plot(future_dates, future_predictions_gelu, label='GELU Future Prediction', color='red', linestyle=':')

    plt.title(f'Future Prediction Comparison (Next {future_days} Days)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("✅ Analysis complete! Models have been trained, evaluated, and compared.")

if __name__ == "__main__":
    main()
