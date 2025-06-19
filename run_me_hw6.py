from abc import ABC, abstractmethod
import os
import random
from typing import Dict, Tuple
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier
from assets.DataProvider import DataProvider
from assets.FeaturesGenerator import FeaturesGenerator
from assets.enums import DataPeriod, DataResolution
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

class CNNClassifier:
    """CNN classifier that follows scikit-learn's estimator interface"""
    
    def __init__(self, input_shape: int = None, random_state: int = None,
                 epochs: int = 10, batch_size: int = 32,
                 conv1_filters: int = 32, conv2_filters: int = 64,
                 dense_units: int = 64, dropout_rate: float = 0.5):
        self.input_shape = input_shape
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.classes_ = None
        self.model = None
        
    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'input_shape': self.input_shape,
            'random_state': self.random_state,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'conv1_filters': self.conv1_filters,
            'conv2_filters': self.conv2_filters,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate
        }
    
    def set_params(self, **params) -> 'CNNClassifier':
        """Set the parameters of this estimator.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f'Invalid parameter {key} for estimator {self.__class__.__name__}')
            setattr(self, key, value)
        return self
        
    def _build_model(self):
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
            
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape((self.input_shape, 1), input_shape=(self.input_shape,)),
            tf.keras.layers.Conv1D(self.conv1_filters, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(self.conv2_filters, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.dense_units, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    @property
    def _estimator_type(self):
        return "classifier"
    
    def fit(self, X, y):
        """Fit the model to the data"""
        # Initialize model if not already done
        if self.model is None:
            self.model = self._build_model()
            
        # Store unique classes
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("CNNClassifier only supports binary classification")
            
        # Convert labels to 0/1
        y_binary = (y == self.classes_[1]).astype('float32')
        
        # Fit the model
        self.model.fit(
            X, y_binary,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        probs = self.model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """Predict class labels"""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

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

# Initialize data provider
data_provider = DataProvider(
    tickers=['BTC/USDT'],
    resolution=DataResolution.DAY_01,
    period=DataPeriod.YEAR_01
)

# Load and process data
data_provider.data_load()
#data_provider.clean_data()

# Get processed data and prepare features
df = data_provider.data_processed['BTC/USDT']

# Generate labels first
y = (df['Returns'].shift(-1) > 0).astype(int)

# Generate features
features_generator = FeaturesGenerator()
features, feature_names = features_generator.prepare_features(df)

# Align the data - use the same index as features
y = y[features.index]

# Convert to numpy arrays
X = features.values
y = y.values

# Remove the last row since we can't predict it
X = X[:-1]
y = y[:-1]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Print shapes for debugging
print(f"X shape: {X_scaled.shape}, y shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=SEED)

# Use RobustScaler for better handling of outliers in financial data
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X_scaled)

# Create optimized base models focusing on strong performers
base_models = {
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        bootstrap=True,
        random_state=SEED
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=SEED
    ),
    'CNN': CNNClassifier(
        input_shape=X_robust.shape[1],
        random_state=SEED,
        conv1_filters=128,
        conv2_filters=256,
        dense_units=128,
        dropout_rate=0.4
    )
}

# Create stacking ensemble with optimized meta-classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', base_models['XGBoost']),
        ('rf', base_models['Random Forest']),
        ('gb', base_models['Gradient Boosting']),
        ('cnn', base_models['CNN'])
    ],
    final_estimator=XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=SEED
    ),
    cv=5,
    n_jobs=-1
)

# Train all models
model_metrics = {}
evaluator = ModelEvaluator()

# Train and evaluate base models
print("Training base models...")
for name, model in base_models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    metrics = evaluator.evaluate_model(model, X_test, y_test)
    model_metrics[name] = metrics
    print(f"{name} performance: {metrics}")

# Train and evaluate stacking ensemble
print("\nTraining Stacking Ensemble...")
stacking_clf.fit(X_train, y_train)
model_metrics['Stacking Ensemble'] = evaluator.evaluate_model(stacking_clf, X_test, y_test)
print(f"Stacking Ensemble performance: {model_metrics['Stacking Ensemble']}")

# Print metrics
print("\nModel Performance Metrics:")
for model_name, metrics in model_metrics.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# Plot performance comparison
evaluator.plot_model_comparison(model_metrics)
