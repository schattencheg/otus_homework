#%pip install backtesting
#%pip install nbformat
import os
import traceback

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from assets.DataProvider import DataProvider
from assets.FeaturesGenerator import FeaturesGenerator
from assets.hw5.HybridStrategySMA_ML import HybridStrategySMA_ML
from assets.hw5.Models import StockCNN, StockCNN_LSTM, StockGRU, StockLSTM
from assets.hw5.SMABaselineGenerator import SMABaselineGenerator
from backtesting import Backtest, Strategy


def main(ticker='BTC/USDT', timeframe='1h'):
    # 0. Initial parameters
    best_params_sma = {} # Initialize to handle cases where SMA optimization might be skipped
    ticker = ticker
    timeframe = timeframe
    threshold = 0.2
    # 1. Load the data
    data_provider = DataProvider(tickers=[ticker], skip_dashboard=True)
    if ticker not in data_provider.data:
        if not data_provider.data_load():
            data_provider.data_request()
            data_provider.data_save()
        data_provider.clean_data()
    data = data_provider.data[ticker]
    # 2. Add features
    features_generator = FeaturesGenerator()
    data_initial_ohlc = data.copy()  # Store a true copy of original OHLC data
    feature_matrix, feature_names = features_generator.prepare_features(data_initial_ohlc) # Pass original; 'data' is now data_ohlc_aligned_with_features
    num_features = feature_matrix.shape[1]
    print(f"Number of features generated: {num_features} ({len(feature_names)} names)")
    # 3. Generate target
    # 3.1. Generate trades, using homework 2 (only long)
    strategy_hw2 = SMABaselineGenerator(ticker, timeframe, data_provider)
    strategy_hw2.evaluate_strategies()
    trades_sma = strategy_hw2.best_trades
    params_sma = strategy_hw2.best_params
    # 3.2. Extract data for ML
    # Add PnL and Returns from strategy run as additional features
    data_extended = feature_matrix.copy() 
    if trades_sma is None or trades_sma.empty:
        print("WARN: No trades from SMA strategy. PnL and ReturnPct will be zero for y_base generation.")
        data_extended['PnL'] = 0.0
        data_extended['ReturnPct'] = 0.0
    else:
        trades_sma = trades_sma.set_index('EntryTime')
        # Merge PnL and ReturnPct from trades_sma into data_extended
        # Use a left merge to keep all rows from data_extended (which is aligned with features)
        # and fill missing PnL/ReturnPct with 0 for non-trade days or if trades_sma is shorter
        data_extended = data_extended.merge(trades_sma[['PnL', 'ReturnPct']], 
                                            left_index=True, right_index=True, how='left')
        data_extended[['PnL', 'ReturnPct']] = data_extended[['PnL', 'ReturnPct']].fillna(value=0)
    trades_return_pct = data_extended['ReturnPct'].values
    y_base = [1 if x > threshold else 0 for x in trades_return_pct]
    # 4. Dataset generation
    print(f"DEBUG: Shape of feature_matrix before create_sequences: {feature_matrix.shape}")
    print(f"DEBUG: Length of y_base before create_sequences: {len(y_base)}")
    window_size = 30  # Define window_size
    # Ensure feature_matrix.values and y_base are used with the create_sequences function
    X_np, y_np = create_sequences(feature_matrix.values, np.array(y_base), window_size)
    print(f"DEBUG: Shape of X_np after create_sequences: {X_np.shape}")
    print(f"DEBUG: Shape of y_np after create_sequences: {y_np.shape}")
    # Now convert lists to tensors
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)  # Use y_np here
    # DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # ... (rest of the code remains the same)
    models = {'CNN': {  'model': StockCNN(input_channels=num_features, window_size=window_size), 
                        'title': 'CNN model Predictions vs. Actual',
                        'model_trained': None},
              'LSTM': { 'model': StockLSTM(input_size=num_features), 
                        'title': 'LSTM model Predictions vs. Actual',
                        'model_trained': None},
              'GRU': {  'model': StockGRU(input_size=num_features), 
                        'title': 'GRU model Prediction vs. Actual',
                        'model_trained': None},
              'CNN-LSTM': { 'model': StockCNN_LSTM(input_channels=num_features), 
                            'title': 'CNN-LSTM model Prediction vs. Actual',
                            'model_trained': None}}
    
    models = {'CNN': {  'model': StockCNN(input_channels=num_features, window_size=window_size), 
                        'title': 'CNN model Predictions vs. Actual',
                        'model_trained': None}}
    
    # 6. Train models
    for model in models:
        print(f"Training {model} model...")
        model_trained = train_model(models[model]['title'], models[model]['model'], dataloader)
        models[model]['model_trained'] = model_trained
    # 7. Test models
    for model in models:
        test_model(models[model]['title'], models[model]['model_trained'], dataloader)
    # 8. Save models
    for model in models:
        if not os.path.exists(f"hw5"):
            os.makedirs(f"hw5")
        torch.save(models[model]['model_trained'].state_dict(), f"hw5/{model}_model_trained.pt")
    # 9. Run models over real strategy
    print("\n--- Running Backtests with Trained Models ---")
    all_backtest_results = {} # Initialize dictionary to store results
    for model_key in models: # Renamed loop variable from 'model' for clarity
        print(f"\n\n\nPreparing backtest for model: {model_key}...")
        # HW2Strategy_SMA (in assets/StrategyCollection.py) will need to be modified 
        # to accept and use model, window_size, feature_names, and scaler parameters.
        # For now, instantiating with original SMA parameters to allow the script to run.
        # This backtest will use SMA logic, NOT the trained model.
        print(f"INFO: HW2Strategy_SMA (defined in this file) needs update for model: {model_key}.")
        print("INFO: Running backtest with default SMA logic for now.")
        # The 'data' DataFrame passed to Backtest is the one after 'features_generator.prepare_features(data)'
        # It must contain 'Open', 'High', 'Low', 'Close', 'Volume' and all feature columns.
        print(f"Running backtest for {model_key} using SMA logic (model not integrated yet).")
        # Ensure 'data' is the DataFrame with features included
        # Pass HW2Strategy_SMA class to Backtest constructor
        bt = Backtest(data.copy(),  # Pass data.copy()
                      HybridStrategySMA_ML, 
                      cash=10000000, 
                      commission=.002)
        try:
            # Pass sma_short and sma_long to bt.run()
            stats = bt.run(
                sma_short=params_sma.get('sma_short', HybridStrategySMA_ML.sma_short),
                sma_long=params_sma.get('sma_long', HybridStrategySMA_ML.sma_long),
                model=models[model_key]['model_trained']
            )
            print(f"--- Backtest Results for {model_key} driven strategy ---")
            print(stats)
            all_backtest_results[model_key] = {'stats': stats}
            plot_filename = f"backtest_plot_{model_key}.html"
            try:
                bt.plot(filename=plot_filename, open_browser=False)
                all_backtest_results[model_key]['plot_file'] = plot_filename
                print(f"Backtest plot for {model_key} saved to {plot_filename}")
            except Exception as e:
                print(f"ERROR: Could not generate plot for {model_key}: {e}")
                all_backtest_results[model_key]['plot_file'] = None
        except Exception as e:
            print(f"ERROR during backtest for {model_key}: {e}")
            print("This might be due to HW2Strategy_SMA not being fully adapted for model-based trading,")
            print("or issues with data slicing/feature access within the strategy's next() method.")
            traceback.print_exc() # Print detailed traceback
    print("\n\n--- Summary of All Backtest Results ---")
    for model_key_summary, results_summary in all_backtest_results.items(): # Renamed loop variables for clarity
        print(f"\n--- Results for {model_key_summary} ---")
        print(results_summary['stats'])
        if 'plot_file' in results_summary and results_summary['plot_file']:
            print(f"Plot saved to: {results_summary['plot_file']}")
        else:
            print("Plot generation failed or was not attempted.")

def train_model(title, model, dataloader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()
    return model
    #
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.numpy())
            all_targets.extend(targets.numpy())

    # Plot actual vs predicted labels
    plt.figure(figsize=(12, 6))
    plt.plot(all_targets[-30:], label='Actual', color='blue', linestyle='--', alpha=0.7)
    plt.plot(all_predictions[-30:], label='Predicted', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Signal (0=Sell, 1=Hold, 2=Buy)")
    plt.legend()
    plt.show()

def test_model(title, model, dataloader):
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.numpy())
            all_targets.extend(targets.numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"{title} Accuracy: {accuracy:.4f}")
    return accuracy
    # Plot actual vs predicted labels
    plt.figure(figsize=(12, 6))
    plt.plot(all_targets[-30:], label='Actual', color='blue', linestyle='--', alpha=0.7)
    plt.plot(all_predictions[-30:], label='Predicted', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Signal (0=Sell, 1=Hold, 2=Buy)")
    plt.legend()
    plt.show()
    
    return accuracy

def create_sequences(data_values, target_values, window_size):
    X, y = [], []
    for i in range(len(data_values) - window_size):
        X.append(data_values[i:(i + window_size)])
        y.append(target_values[i + window_size])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    ticker = 'BTC/USDT'
    timeframe = '1h'
    main(ticker, timeframe)
