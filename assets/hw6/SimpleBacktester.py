import pandas as pd
import torch
from tqdm import tqdm


class SimpleBacktester:
    def __init__(self, data, features, lstm_model, cnn_model, deep_cnn_model, voting_model):
        self.data = data
        self.features = features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Move models to device
        self.lstm_model = lstm_model.to(self.device)
        self.cnn_model = cnn_model.to(self.device)
        self.deep_cnn_model = deep_cnn_model.to(self.device)
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
        if signal_strength == 0:
            return 0
        
        # Get current market context
        current_time = self.data.index[-1]
        volatility_regime = self.features['volatility_regime'].iloc[-1]
        trend_strength = self.features['trend_strength'].iloc[-1]
        
        # Calculate portfolio metrics
        portfolio_value = self.cash + sum(pos['size'] * current_price for pos in self.positions)
        current_exposure = sum(pos['size'] * current_price for pos in self.positions) / portfolio_value
        
        # Dynamic risk per trade based on market conditions
        base_risk = 0.02  # Base risk of 2%
        
        # Adjust risk based on market regime
        if volatility_regime == 'high':
            risk_multiplier = 0.7  # Reduce risk in high volatility
        elif volatility_regime == 'low':
            risk_multiplier = 1.2  # Increase risk in low volatility
        else:
            risk_multiplier = 1.0
        
        # Adjust risk based on trend strength
        if trend_strength > 25:
            risk_multiplier *= 1.1  # Increase risk in strong trends
        elif trend_strength < 15:
            risk_multiplier *= 0.9  # Decrease risk in weak trends
        
        # Adjust risk based on current exposure
        if current_exposure > 0.5:
            risk_multiplier *= 0.8  # Reduce risk when heavily invested
        
        # Calculate final risk percentage
        risk_per_trade = base_risk * risk_multiplier
        risk_amount = portfolio_value * risk_per_trade
        
        # Dynamic stop loss calculation
        atr = self.atr.loc[current_time]
        volatility_stop = atr * 2.0  # Base stop at 2 ATR
        
        # Adjust stops based on volatility regime
        if volatility_regime == 'high':
            volatility_stop *= 1.5  # Wider stops in high volatility
        elif volatility_regime == 'low':
            volatility_stop *= 0.8  # Tighter stops in low volatility
        
        # Calculate position size based on risk and stop distance
        size = (risk_amount / volatility_stop) * signal_strength
        
        # Position size constraints
        max_position_value = portfolio_value * 0.25  # Max 25% of portfolio per position
        size = min(size, max_position_value / current_price)
        
        # Adjust for remaining cash (leave buffer for fees)
        available_cash = self.cash * 0.95  # Leave 5% buffer
        size = min(size, available_cash / current_price)
        
        # Scale based on conviction
        if signal_strength > 0.8:
            size *= 1.2  # Increase size for high conviction trades
        elif signal_strength < 0.6:
            size *= 0.8  # Decrease size for lower conviction trades
        
        # Minimum position constraints
        min_trade_value = 1000  # Minimum trade size in USD
        min_size = min_trade_value / current_price
        
        # Final position size
        size = max(min_size, min(size, self.max_position_size))
        
        # Round to 6 decimal places for crypto
        size = round(size, 6)
        
        return size
        
    def run(self):
        self.trades = []
        self.positions = []
        
        # Calculate volatility and ATR
        volatility = self.calculate_volatility()
        self.atr = self._calculate_atr(self.data)
        
        # Initialize portfolio value
        self.portfolio_value = self.cash
        
        # Use tqdm for progress bar
        for i in tqdm(range(self.window_size, len(self.data)), desc='Backtesting'):
            current_data = self.data.iloc[:i+1]
            current_price = current_data['Close'].iloc[-1]
            current_time = current_data.index[-1]
            current_vol = volatility.iloc[i] if i < len(volatility) else volatility.iloc[-1]
            
            # Update portfolio value
            positions_value = sum(pos['size'] * current_price for pos in self.positions)
            self.portfolio_value = self.cash + positions_value
            
            # Check stops and handle position exits
            closed_positions = self.check_stops(current_data)
            
            # Get trading signal
            signal = self._get_signal(current_data, self.features.iloc[:i+1])
            
            # Skip if max positions reached
            if len(self.positions) >= self.max_positions:
                continue
            
            # Calculate position size if signal exists
            if signal > 0:
                # Calculate dynamic stop levels
                stop_loss_pct, take_profit_pct = self.get_dynamic_stops(current_price, current_time)
                
                # Calculate position size
                position_size = self.calculate_position_size(signal, current_price, current_vol)
                position_value = position_size * current_price
                
                # Apply position limits
                max_trade_value = min(
                    self.cash * 0.3,  # Max 30% of cash
                    self.portfolio_value * 0.2,  # Max 20% of portfolio
                    self.max_position_size * current_price  # Max BTC position
                )
                position_value = min(position_value, max_trade_value)
                
                # Skip if position too small
                if position_value < 1000:  # Minimum $1000 position
                    continue
                
                # Skip if not enough cash
                if position_value > self.cash:
                    continue
                
                # Calculate final position size
                position_size = position_value / current_price
                
                # Print trade details
                print(f"\nOpening position at {current_time}")
                print(f"Price: ${current_price:.2f}, Size: {position_size:.3f} BTC")
                print(f"Stop Loss: ${current_price * (1 - stop_loss_pct):.2f}")
                print(f"Take Profit: ${current_price * (1 + take_profit_pct):.2f}")
                
                # Open position
                self.positions.append({
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'size': position_size,
                    'stop_price': current_price * (1 - stop_loss_pct),
                    'take_profit': current_price * (1 + take_profit_pct),
                    'high_water_mark': current_price,
                    'trailing_stop': current_price * (1 - self.trailing_stop_pct),
                    'entry_atr': self.atr.iloc[i],
                    'entry_vol': current_vol,
                    'initial_stop': current_price * (1 - stop_loss_pct),
                    'highest_price': current_price
                })
                
                # Update cash
                self.cash -= current_price * position_size * (1 + self.commission)
                
                # Update peak value if new high
                if self.portfolio_value > self.peak_value:
                    self.peak_value = self.portfolio_value
                
                # Update max drawdown
                drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
                self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Close any remaining positions
        trades = self.close_positions(self.data)
        
        # Print final statistics
        print("\nBacktest Complete")
        print(f"Final Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Return: {((self.portfolio_value / self.initial_cash) - 1) * 100:.2f}%")
        print(f"Max Drawdown: {self.max_drawdown * 100:.2f}%")
        
        return trades
        
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
                self.max_position_size * current_price  # Max BTC position
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
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices):
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices, period=20, std=2):
        middle = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    def _calculate_ema(self, prices, period):
        return prices.ewm(span=period, adjust=False).mean()


    def check_stops(self, data):
        current_time = data.index[-1]
        current_price = data['Close'].iloc[-1]
        
        # Update portfolio metrics
        portfolio_value = self.cash + sum(pos['size'] * current_price for pos in self.positions)
        self.peak_value = max(self.peak_value, portfolio_value)
        
        # Calculate drawdown
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value * 100
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Get market context
        volatility_regime = self.features['volatility_regime'].iloc[-1]
        trend_strength = self.features['trend_strength'].iloc[-1]
        current_atr = self.atr.loc[current_time]
        
        # Portfolio-wide risk check
        if current_drawdown > 15:  # Maximum drawdown threshold
            self._close_all_positions(current_time, current_price, 'max_drawdown')
            return
        
        # Check each position
        for position in self.positions:
            # Calculate unrealized P&L
            unrealized_pnl = (current_price - position['entry_price']) / position['entry_price']
            time_in_trade = (current_time - position['entry_time']).total_seconds() / 3600  # hours
            
            # Dynamic stop loss adjustment
            initial_stop = position['initial_stop']
            trailing_stop = position['trailing_stop']
            
            # Adjust stops based on profit and time
            if unrealized_pnl > 0.02:  # In profit > 2%
                # Tighten stops in profit
                trailing_stop = max(trailing_stop * 0.8, initial_stop * 0.5)
                
                # Time-based stop adjustment
                if time_in_trade > 48:  # After 48 hours
                    trailing_stop = max(trailing_stop * 0.7, initial_stop * 0.4)
            
            # Volatility-based stop adjustment
            current_vol = current_atr / current_price
            if current_vol > position['entry_vol'] * 1.5:  # Volatility increased
                trailing_stop *= 1.2  # Widen stops
            elif current_vol < position['entry_vol'] * 0.7:  # Volatility decreased
                trailing_stop *= 0.8  # Tighten stops
            
            # Update trailing stop if price moved in our favor
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
                new_stop = current_price * (1 - trailing_stop)
                position['stop_price'] = max(position['stop_price'], new_stop)
            
            # Check stop conditions
            stop_triggered = False
            exit_reason = None
            
            # 1. Stop loss hit
            if current_price <= position['stop_price']:
                stop_triggered = True
                exit_reason = 'stop_loss'
            
            # 2. Take profit hit
            elif current_price >= position['entry_price'] * (1 + position['take_profit']):
                stop_triggered = True
                exit_reason = 'take_profit'
            
            # 3. Trend reversal exit
            elif time_in_trade > 24:  # Only check after 24 hours
                if trend_strength < 15 and unrealized_pnl > 0:
                    stop_triggered = True
                    exit_reason = 'trend_reversal'
            
            # 4. Volatility exit
            if volatility_regime == 'high' and unrealized_pnl > 0.01:
                stop_triggered = True
                exit_reason = 'high_volatility'
            
            # Close position if any stop condition met
            if stop_triggered:
                self._close_position(position, current_time, current_price, exit_reason)
    
    def _close_position(self, position, current_time, current_price, reason):
        """Helper method to close a single position"""
        pnl = (current_price - position['entry_price']) * position['size']
        pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
        
        closed_position = {
            'entry_time': position['entry_time'],
            'exit_time': current_time,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'size': position['size'],
            'pnl': pnl_pct,
            'exit_reason': reason,
            'entry_atr': position['entry_atr'],
            'entry_vol': position['entry_vol'],
            'time_in_trade': (current_time - position['entry_time']).total_seconds() / 3600
        }
        
        self.trades.append(closed_position)
        self.cash += current_price * position['size'] * (1 - self.commission)
        if position in self.positions:
            self.positions.remove(position)
        
        # Update performance metrics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        
        print(f"{reason.upper()}: Closing position from {position['entry_time']} at {current_time}")
        print(f"Entry: ${position['entry_price']:.2f}, Exit: ${current_price:.2f}")
        print(f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
    
    def _close_all_positions(self, current_time, current_price, reason):
        """Helper method to close all positions"""
        positions_copy = self.positions.copy()
        for position in positions_copy:
            self._close_position(position, current_time, current_price, reason)
            
            # Clear positions
            self.positions = []
        
        return pd.DataFrame(self.trades)

    def close_positions(self, data):
        # Close any remaining positions at the end
        if self.positions:
            final_price = data['Close'].iloc[-1]
            final_time = data.index[-1]
            
            for position in self.positions:
                pnl = (final_price - position['entry_price']) * position['size']
                pnl_pct = (final_price - position['entry_price']) / position['entry_price'] * 100
                
                closed_position = {
                    'entry_time': position['entry_time'],
                    'exit_time': final_time,
                    'entry_price': position['entry_price'],
                    'exit_price': final_price,
                    'size': position['size'],
                    'pnl': pnl_pct,
                    'exit_reason': 'end_of_period',
                    'entry_atr': position['entry_atr'],
                    'entry_vol': position['entry_vol']
                }
                
                self.trades.append(closed_position)
                self.cash += final_price * position['size'] * (1 - self.commission)
                
                # Print trade details
                print(f"END_OF_PERIOD: Closing position from {position['entry_time']} at {final_time}")
                print(f"Entry: ${position['entry_price']:.2f}, Exit: ${final_price:.2f}")
                print(f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Clear positions
            self.positions = []
        
        return pd.DataFrame(self.trades)

    def _get_signal(self, data, features):
        # Market regime analysis
        volatility_regime = features['volatility_regime'].iloc[-1]
        trend_strength = features['trend_strength'].iloc[-1]
        trend_direction = features['trend_direction'].iloc[-1]
        
        # Technical signals
        rsi_signal = (
            features['rsi_14'].iloc[-1],
            features['rsi_14_slope'].iloc[-1],
            features['rsi_14_ma'].iloc[-1]
        )
        
        macd_signal = (
            features['macd'].iloc[-1],
            features['macd_signal'].iloc[-1],
            features['macd_hist'].iloc[-1],
            features['macd_hist_slope'].iloc[-1]
        )
        
        # Volume analysis
        volume_signals = (
            features['volume_ratio'].iloc[-1],
            features['volume_price_trend'].iloc[-1],
            features['volume_price_corr'].iloc[-1],
        )
        
        # Price action patterns
        price_patterns = (
            features['body_size'].iloc[-1],
            features['upper_shadow'].iloc[-1],
            features['lower_shadow'].iloc[-1],
            features['high_low_range'].iloc[-1]
        )
        
        # Get model predictions with market regime context
        with torch.no_grad():
            self.lstm_model.eval()
            self.cnn_model.eval()
            self.voting_model.eval()
            
            # Prepare sequence data
            seq_data = features.iloc[-self.window_size:].values
            
            # LSTM prediction
            lstm_input = torch.FloatTensor(seq_data).unsqueeze(0).to(self.device)
            lstm_pred = self.lstm_model(lstm_input)
            
            # CNN prediction
            cnn_input = torch.FloatTensor(seq_data).unsqueeze(0).transpose(1, 2).to(self.device)
            cnn_pred = self.cnn_model(cnn_input)
            
            # Voting model prediction
            voting_input = torch.FloatTensor(seq_data[-1]).unsqueeze(0).to(self.device)
            voting_pred = self.voting_model(voting_input)
            
            # Get deep CNN prediction
            deep_cnn_pred = self.deep_cnn_model(cnn_input)
            
            # Dynamic model weighting based on market regime
            if volatility_regime == 'high':
                # In high volatility, trust LSTM and Deep CNN more
                weights = (0.35, 0.15, 0.35, 0.15)  # LSTM, CNN, Deep CNN, Voting
            elif volatility_regime == 'low':
                # In low volatility, trust voting and CNN models more
                weights = (0.2, 0.25, 0.2, 0.35)  # LSTM, CNN, Deep CNN, Voting
            else:
                # In medium volatility, balanced weights with slight preference to Deep CNN
                weights = (0.25, 0.2, 0.3, 0.25)  # LSTM, CNN, Deep CNN, Voting
            
            # Calculate regime-aware ensemble prediction
            ensemble_pred = (
                lstm_pred.softmax(dim=1) * weights[0] +
                cnn_pred.softmax(dim=1) * weights[1] +
                deep_cnn_pred.softmax(dim=1) * weights[2] +
                voting_pred.softmax(dim=1) * weights[3]
            )
            
            confidence = ensemble_pred[0][1].item()
            
            self.lstm_model.train()
            self.cnn_model.train()
            self.deep_cnn_model.train()
            self.voting_model.train()
        
        # Signal strength calculation based on multiple factors
        signal_strength = 0.0
        
        # 1. Technical Analysis Score (30%)
        tech_score = 0.0
        # RSI conditions
        if 30 <= rsi_signal[0] <= 70:  # Not overbought/oversold
            tech_score += 0.3
            if rsi_signal[1] > 0 and rsi_signal[0] > rsi_signal[2]:  # Positive slope and above MA
                tech_score += 0.2
        
        # MACD conditions
        if macd_signal[2] > 0 and macd_signal[3] > 0:  # Positive histogram and slope
            tech_score += 0.3
        if macd_signal[0] > macd_signal[1]:  # MACD above signal
            tech_score += 0.2
        
        # 2. Volume Analysis Score (20%)
        vol_score = 0.0
        if volume_signals[0] > 1.0:  # Above average volume
            vol_score += 0.4
        if volume_signals[1] > 0:  # Positive volume trend
            vol_score += 0.3
        
        # 3. Price Action Score (20%)
        price_score = 0.0
        if price_patterns[0] > 0.001:  # Significant body size
            price_score += 0.3
        if price_patterns[2] < price_patterns[1]:  # Lower shadow < upper shadow (bullish)
            price_score += 0.3
        if price_patterns[3] < 0.02:  # Not too volatile
            price_score += 0.4
        
        # 4. Model Confidence (30%)
        model_score = confidence
        
        # Combine scores with weights
        signal_strength = (
            tech_score * 0.3 +
            vol_score * 0.2 +
            price_score * 0.2 +
            model_score * 0.3
        )
        
        # Apply market regime filters
        if volatility_regime == 'high':
            signal_strength *= 0.7  # Reduce position size in high volatility
        if trend_strength < 20:  # Weak trend
            signal_strength *= 0.8
        if trend_direction == -1:  # Counter-trend
            signal_strength *= 0.6
        
        # Dynamic thresholds based on market conditions
        base_threshold = 0.5
        if volatility_regime == 'high':
            base_threshold = 0.6
        elif volatility_regime == 'low':
            base_threshold = 0.45
        
        # Final signal decision
        if signal_strength > base_threshold + 0.2:
            print(f"Strong signal: {signal_strength:.2f}")
            return min(1.0, signal_strength)
        elif signal_strength > base_threshold:
            print(f"Moderate signal: {signal_strength:.2f}")
            return min(0.75, signal_strength)
        elif signal_strength > base_threshold - 0.1:
            print(f"Weak signal: {signal_strength:.2f}")
            return min(0.5, signal_strength)
        
        print(f"No signal: {signal_strength:.2f}")
        return 0.0
