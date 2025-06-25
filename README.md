# CNN-Based Crypto Trading Strategy

This project implements a CNN-based trading strategy for cryptocurrency markets, specifically for BTC/USDT trading. Designed to run in Docker containers on cloud VMs.

## Components

1. `run_me_hw7.py` - Main training script that:
   - Loads historical data using DataProvider
   - Generates technical features
   - Trains a CNN model
   - Backtests the strategy
   - Saves the model and performance plots

2. `live_trading.py` - Live trading script that:
   - Loads a trained model
   - Connects to Binance Futures testnet
   - Executes trades based on model predictions
   - Logs all activities and performance

## Docker Setup

1. Install Docker and Docker Compose on your cloud VM:
```bash
curl -fsSL https://get.docker.com | sh
sudo apt-get install docker-compose
```

2. Set up Binance API credentials:
   - Create a Binance account and generate API keys
   - Enable futures trading
   - Create a `.env` file:
```bash
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

## Usage with Docker

1. Train and backtest the model:
```bash
# Run with Docker
docker-compose up backtest

# Or run directly
python run_me_hw7.py backtest
```
This will:
- Load historical BTC/USDT data
- Train the CNN model
- Save the model to `models/cnn_trader.pth`
- Save performance plots to `data/strategy_performance.png`

2. Run live trading:
```bash
# Run with Docker
docker-compose up -d trader

# Or run directly
python run_me_hw7.py live
```
This will:
- Load the trained model
- Connect to Binance Futures testnet
- Start trading with 0.01 BTC position size
- Log all activities to `data/trading.log`

3. Monitor trading logs:
```bash
docker-compose logs -f trader
# or
tail -f data/trading.log
```

## Directory Structure

```
./
├── models/           # Saved model files
├── data/            # Data, logs, and plots
├── assets/          # Project modules
├── Dockerfile       # Container definition
├── docker-compose.yml # Service configuration
├── run_me_hw7.py    # Training script
└── live_trading.py  # Trading script
```

## Configuration

- Model parameters: Modify `run_me_hw7.py`
- Trading parameters: Adjust `live_trading.py`
- Container settings: Edit `docker-compose.yml`
- Data persistence: Both `models` and `data` directories are mounted as volumes

## Cloud VM Requirements

- Minimum 2 CPU cores
- 4GB RAM
- 20GB disk space
- Ubuntu 20.04 or later recommended
- Open outbound ports for Binance API access

## Monitoring and Maintenance

1. Check container status:
```bash
docker-compose ps
```

2. View logs:
```bash
docker-compose logs -f
```

3. Restart trading container:
```bash
docker-compose restart trader
```

4. Stop all containers:
```bash
docker-compose down
```

## Notes

- The live trading script uses Binance Futures testnet by default
- Containers will automatically restart on failure
- All data is persisted in mounted volumes
- Monitor system resources and logs regularly
- Consider setting up monitoring alerts for critical events



The key changes made for Docker/cloud VM compatibility:

Modified scripts to:
Save plots to files instead of displaying them
Use non-interactive matplotlib backend
Add proper logging with file output
Create necessary data directories
Handle paths correctly for container environment
Added Docker configuration:
Dockerfile with Python environment setup
docker-compose.yml with separate services for training and trading
Volume mounts for persistent data and models
Environment variable handling
Automatic container restart
Added cloud VM considerations:
Resource requirements
Data persistence
Logging and monitoring
Container management commands
To deploy on a cloud VM:

Copy all files to the VM
Create .env file with Binance API credentials
Run docker-compose up trainer to train the model
Run docker-compose up -d trader to start trading
Monitor logs with docker-compose logs -f trader
The system will automatically handle crashes and restarts, and all data is persisted in mounted volumes.