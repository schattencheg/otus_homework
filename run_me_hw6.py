import os
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import backtesting
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier,
                            StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# Set environment variable for deterministic PyTorch behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Set seed for reproducibility
SEED = 777

class DataLoaderBase(ABC):
    """Base class for data loading interface"""
    
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data and return features and labels"""
        pass

class SyntheticDataLoader(DataLoaderBase):
    """Load synthetic data for ML experiments"""
    
    def __init__(self, n_samples: int = 1000, n_features: int = 4,
                 n_informative: int = 2, n_redundant: int = 2,
                 random_state: int = None):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.random_state = random_state
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            random_state=self.random_state,
            shuffle=False
        )
        return X, y

class ModelEvaluator:
    """Evaluate and compare model performances"""
    
    @staticmethod
    def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Calculate various metrics for model evaluation"""
        y_pred = model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
    
    @staticmethod
    def plot_model_comparison(model_metrics: Dict[str, Dict[str, float]]) -> None:
        """Plot performance comparison of different models"""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = list(model_metrics.keys())
        
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, model_name in enumerate(models):
            performance = [model_metrics[model_name][metric] for metric in metrics]
            ax.bar(x + i * width, performance, width, label=model_name)
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def seed_everything(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print(f"Using {seed} seed")

# Set the seed
seed_everything(SEED)

# Initialize data loader
data_loader = SyntheticDataLoader(
    n_samples=1_000,
    n_features=4,
    n_informative=2,
    n_redundant=2,
    random_state=SEED
)

# Load and split data
X, y = data_loader.load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)

# Create base models
base_models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=SEED),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=SEED)
}

# Create voting ensemble
voting_clf = VotingClassifier(estimators=[
    ('lr', base_models['Logistic Regression']),
    ('rf', base_models['Random Forest']),
    ('gb', base_models['Gradient Boosting'])], voting='hard')

# Train all models
model_metrics = {}
evaluator = ModelEvaluator()

# Train and evaluate base models
for name, model in base_models.items():
    model.fit(X_train, y_train)
    model_metrics[name] = evaluator.evaluate_model(model, X_test, y_test)

# Train and evaluate ensemble
voting_clf.fit(X_train, y_train)
model_metrics['Voting Ensemble'] = evaluator.evaluate_model(voting_clf, X_test, y_test)

# Print metrics
print("\nModel Performance Metrics:")
for model_name, metrics in model_metrics.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# Plot performance comparison
evaluator.plot_model_comparison(model_metrics)
