import pandas as pd
import torch
from tqdm import tqdm


class SimpleBacktester:
    def __init__(self, data, features, lstm_model, cnn_model, voting_model):
        self.data = data
        self.features = features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Move models to device
        self.lstm_model = lstm_model.to(self.device)
        self.cnn_model = cnn_model.to(self.device)
        self.voting_model = voting_model.to(self.device)
        
        # Initialize portfolio
        self.initial_cash = 100000  # Starting with $100k
        self.cash = self.initial_cash
        self.positions = []  # List to track open positions
        self.trades = []  # List to track completed trades
        
        # Risk parameters
        self.max_positions = 3  # Max number of concurrent positions
        self.max_position_size = 0.5  # Maximum position size in BTC
        self.base_stop_loss_pct = 0.015  # Base stop loss at 1.5%
        self.trailing_stop_pct = 0.008  # 0.8% trailing stop
        self.risk_per_trade = 0.02  # Risk 2% per trade
        self.commission = 0.001  # 0.1% commission per trade
        
        # Performance tracking
        self.peak_value = 100000
        self.max_drawdown = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Technical parameters
        self.window_size = 60  # Lookback window for features
        
        # Initialize performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0
        
        # Calculate ATR for dynamic stop losses
        self.atr = self.calculate_atr(14)
        
        # Initialize performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0
        self.peak_value = self.cash
    
    def calculate_atr(self, period):
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window=period).mean()
    
    def get_dynamic_stops(self, current_price, current_time):
        # Get current ATR value
        current_atr = self.atr.loc[current_time]
        atr_multiplier = 2.0
        
        # Calculate dynamic stop loss (2-3 ATR)
        stop_loss_pct = min(self.base_stop_loss_pct * 1.5,
                           (current_atr * atr_multiplier) / current_price)
        
        # Calculate dynamic take profit (1.5-2x stop loss)
        take_profit_pct = stop_loss_pct * 2
        
        return stop_loss_pct, take_profit_pct
    
    def calculate_volatility(self, window=20):
        returns = self.data['Close'].pct_change()
        return returns.rolling(window=window).std()
        
    def calculate_position_size(self, signal_strength, current_price, current_vol):
        # Base position size as percentage of portfolio
        risk_per_trade = 0.03  # 3% risk per trade
        account_value = self.cash + sum(pos['size'] * current_price for pos in self.positions)
        base_size = risk_per_trade * signal_strength
        
        # Adjust for volatility (less impact)
        vol_percentile = current_vol / self.data['Close'].pct_change().std()
        vol_factor = 1 / (1 + vol_percentile * 0.5)  # Less reduction for volatility
        
        # Calculate target position value
        target_value = base_size * account_value * vol_factor
        size = target_value / current_price
        
        # Ensure minimum trade size of $1000
        min_size = 1000 / current_price
        size = max(size, min_size)
        
        # Cap at max position size
        size = min(size, self.max_position_btc)
        
        # Round to 3 decimal places
        return round(size, 3)
        
    def run(self):
        self.trades = []
        self.positions = []
        self.peak_value = self.initial_cash
        self.max_drawdown = 0
        
        # Calculate volatility and other risk metrics
        returns = self.data['Close'].pct_change()
        volatility = returns.rolling(window=20).std()
        atr = self._calculate_atr(self.data)
        
        # Track consecutive losses for dynamic position sizing
        consecutive_losses = 0
        max_consecutive_losses = 3  # Reduce position size after this many losses
        
        print("\nRunning backtest...")
        for i in tqdm(range(self.window_size, len(self.data)), desc="Backtesting"):
            current_data = self.data.iloc[:i+1]
            current_price = current_data['Close'].iloc[-1]
            current_time = current_data.index[-1]
            current_vol = volatility.iloc[i] if i < len(volatility) else volatility.iloc[-1]
            current_atr = atr.iloc[i] if i < len(atr) else atr.iloc[-1]
            
            # Check for stop losses and take profits
            if self.positions:
                closed_positions = self.check_stops(current_data)
                # Update consecutive losses
                for pos in closed_positions:
                    if pos['pnl'] < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
            
            # Update portfolio metrics
            portfolio_value = self.cash + sum(pos['size'] * current_price for pos in self.positions)
            
            # Update peak value and drawdown
            if portfolio_value > self.peak_value:
                self.peak_value = portfolio_value
            else:
                drawdown = (self.peak_value - portfolio_value) / self.peak_value
                self.max_drawdown = max(self.max_drawdown, drawdown)
            
            # Skip if we have max positions or in severe drawdown
            if len(self.positions) >= self.max_positions or drawdown > 0.2:
                continue
            
            # Get trading signal and current features
            signal = self._get_signal(current_data, self.features.iloc[:i+1])
            if signal == 0:
                print("No position taken")
                continue
            
            # Dynamic position sizing based on multiple factors
            volatility_factor = 1 / (1 + current_vol)
            atr_factor = 1 / (1 + current_atr/current_price)  # Normalize ATR by price
            streak_factor = max(0.5, 1 - (consecutive_losses * 0.2))  # Reduce size after losses
            
            # Base position size on portfolio value and risk
            risk_amount = portfolio_value * self.risk_per_trade * streak_factor
            position_value = risk_amount * volatility_factor * atr_factor * signal
            
            # Apply position limits
            max_trade_value = min(
                self.cash * 0.3,  # Max 30% of cash
                portfolio_value * 0.2,  # Max 20% of portfolio
                self.max_position_btc * current_price  # Max BTC position
            )
            position_value = min(position_value, max_trade_value)
            
            # Dynamic stop loss based on ATR
            atr_multiplier = 2
            stop_distance = max(
                current_atr * atr_multiplier,  # ATR-based stop
                current_price * self.base_stop_loss_pct  # Minimum stop distance
            )
            
            # Convert to units and apply minimum size
            position_size = round(position_value / current_price, 6)
            if position_size < 0.001:
                continue
            
            # Calculate stop loss and take profit levels
            stop_loss = current_price - stop_distance
            take_profit = current_price + (stop_distance * 1.5)  # 1.5:1 reward-risk ratio
            
            # Open position with enhanced tracking
            self.positions.append({
                'entry_time': current_time,
                'entry_price': current_price,
                'size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'high_water_mark': current_price,
                'entry_atr': current_atr,
                'entry_vol': current_vol
            })
            
            # Update cash (account for commission)
            self.cash -= current_price * position_size * (1 + self.commission)
        
    def _calculate_atr(self, data, period=14):
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()

    def close_positions(self, data):
        # Close any remaining positions at the end
        if self.positions:
            final_price = data['Close'].iloc[-1]
            final_time = data.index[-1]
            
            for pos in self.positions:
                pnl = (final_price - pos['entry_price']) * pos['size']
                pnl_pct = (final_price - pos['entry_price']) / pos['entry_price'] * 100
                
                self.trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': final_time,
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'size': pos['size'],
                    'pnl': pnl_pct,
                    'exit_reason': 'end_of_period',
                    'entry_atr': pos['entry_atr'],
                    'entry_vol': pos['entry_vol']
                })
                
                self.cash += final_price * pos['size'] * (1 - self.commission)
        
        return pd.DataFrame(self.trades)

    def check_stops(self, data):
        positions_to_close = []
        current_price = data['Close'].iloc[-1]
        current_time = data.index[-1]
        
        for position in self.positions:
            # Update high water mark for trailing stop
            if current_price > position['high_water_mark']:
                position['high_water_mark'] = current_price
                # Update trailing stop to lock in profits
                new_trailing_stop = current_price * (1 - self.trailing_stop_pct)
                if 'trailing_stop' not in position or new_trailing_stop > position['trailing_stop']:
                    position['trailing_stop'] = new_trailing_stop
            
            # Check take profit
            if current_price >= position['take_profit']:
                positions_to_close.append((position, 'take_profit'))
                continue
            
            # Check stop loss
            if current_price <= position['stop_loss']:
                positions_to_close.append((position, 'stop_loss'))
                continue
            
            # Check trailing stop
            if 'trailing_stop' in position and current_price <= position['trailing_stop']:
                positions_to_close.append((position, 'trailing_stop'))
                continue
        
        # Close positions that hit stops
        for position, reason in positions_to_close:
            pnl = (current_price - position['entry_price']) * position['size']
            pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
            
            print(f"{reason.upper()}: Closing position from {position['entry_time']} at {current_time}")
            print(f"Entry: ${position['entry_price']:.2f}, Exit: ${current_price:.2f}")
            print(f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Add to trades list
            self.trades.append({
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'size': position['size'],
                'pnl': pnl_pct,
                'exit_reason': reason
            })
            
            # Update cash (account for commission)
            self.cash += current_price * position['size'] * (1 - self.commission)
            self.positions.remove(position)

    def _get_signal(self, data, features):
        # Get latest features
        current_features = features.iloc[-1]
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.iloc[-1]
        
        # Calculate MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd = macd.iloc[-1]
        macd_signal = signal.iloc[-1]
        
        # Calculate multiple moving averages
        sma_20 = data['Close'].rolling(window=20).mean()
        sma_50 = data['Close'].rolling(window=50).mean()
        ema_9 = data['Close'].ewm(span=9, adjust=False).mean()
        ema_21 = data['Close'].ewm(span=21, adjust=False).mean()
        
        # Calculate Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = data['Close'].rolling(window=bb_period).mean()
        bb_std_dev = data['Close'].rolling(window=bb_period).std()
        bb_upper = bb_middle + (bb_std_dev * bb_std)
        bb_lower = bb_middle - (bb_std_dev * bb_std)
        
        # Calculate ATR for volatility
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]
        
        # Advanced trend indicators
        price_above_sma = data['Close'].iloc[-1] > sma_20.iloc[-1] > sma_50.iloc[-1]  # Strong uptrend
        ema_trend = ema_9.iloc[-1] > ema_21.iloc[-1]  # Short-term momentum
        bb_position = (data['Close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])  # 0 to 1
        
        # Volume trend
        volume_sma = data['Volume'].rolling(window=20).mean()
        volume_trend = data['Volume'].iloc[-1] > volume_sma.iloc[-1]
        
        # Calculate momentum indicators
        returns = data['Close'].pct_change()
        momentum_10 = returns.rolling(window=10).sum().iloc[-1]
        momentum_bullish = (
            rsi > 50 and  # RSI in bullish territory
            macd > macd_signal and  # MACD bullish crossover
            momentum_10 > 0 and  # Positive 10-period momentum
            price_above_sma and  # Price above moving averages
            ema_trend and  # Short-term trend up
            volume_trend and  # Above average volume
            bb_position > 0.5  # Price in upper half of Bollinger Bands
        )
        
        print(f"RSI: {rsi:.2f}, MACD: {macd:.2f}, MACD Signal: {macd_signal:.2f}")
        print(f"BB Position: {bb_position:.2f}, ATR: {atr:.2f}")
        print(f"Momentum bullish: {momentum_bullish}")
        
        # Get model predictions with higher threshold
        with torch.no_grad():
            self.lstm_model.eval()
            self.cnn_model.eval()
            self.voting_model.eval()
            
            # LSTM prediction
            lstm_input = torch.FloatTensor(features.iloc[-self.window_size:].values).unsqueeze(0).to(self.device)
            lstm_pred = self.lstm_model(lstm_input)
            
            # CNN prediction
            cnn_input = torch.FloatTensor(features.iloc[-self.window_size:].values).unsqueeze(0).transpose(1, 2).to(self.device)
            cnn_pred = self.cnn_model(cnn_input)
            
            # Voting model prediction
            voting_input = torch.FloatTensor(current_features.values).unsqueeze(0).to(self.device)
            voting_pred = self.voting_model(voting_input)
            
            # Weighted ensemble prediction (give more weight to LSTM for time series)
            ensemble_pred = (
                lstm_pred.softmax(dim=1) * 0.4 +  # LSTM gets 40% weight
                cnn_pred.softmax(dim=1) * 0.3 +  # CNN gets 30% weight
                voting_pred.softmax(dim=1) * 0.3  # Voting gets 30% weight
            )
            
            # Put models back in training mode
            self.lstm_model.train()
            self.cnn_model.train()
            self.voting_model.train()
        
        # Get confidence from ensemble prediction
        confidence = ensemble_pred[0][1].item()  # Probability of positive class
        print(f"Model confidence: {confidence:.2f}")
        
        # More conservative signal generation
        if price_trend and momentum_bullish:  # Need both confirmations
            if confidence > 0.60:  # Higher threshold
                print("Taking full position")
                return 1.0
            elif confidence > 0.50:
                print("Taking 75% position")
                return 0.75
            elif confidence > 0.45:
                print("Taking 50% position")
                return 0.5
        
        print("No position taken")
        return 0

