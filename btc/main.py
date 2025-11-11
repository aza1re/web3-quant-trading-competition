from AlgorithmImports import *
from alpha import HybridAlpha  # Ensure this matches your alpha.py file

class MyAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2025, 8, 1)
        self.set_cash(100000)
        
        # Set warm-up period for data pre-loading
        self.set_warm_up(timedelta(days=30))
        
        # Debug to confirm import
        self.debug("Imported HybridAlpha successfully")
        
        # BTC pair only
        btc_binance = self.add_crypto("BTCUSDT", Resolution.MINUTE, Market.BINANCE)
        btc_coinbase = self.add_crypto("BTCUSD", Resolution.MINUTE, Market.COINBASE)
        binance_symbol = btc_binance.symbol
        coinbase_symbol = btc_coinbase.symbol
        
        # Set reality models for accurate fees and slippage per security
        # Binance: Approx 0.1% maker/taker (base tier, adjust if using BNB discounts)
        btc_binance.fee_model = CustomFeeModel(self, 0.001)  # 0.1%
        
        # Coinbase: Approx 0.6% maker (mid-tier, adjust based on volume tier)
        btc_coinbase.fee_model = CustomFeeModel(self, 0.006)
        
        # Optional: Add slippage for liquid pairs (0.01%)
        btc_binance.slippage_model = ConstantSlippageModel(0.0001)
        btc_coinbase.slippage_model = ConstantSlippageModel(0.0001)
        
        # Add HybridAlpha model for BTC only
        self.add_alpha(HybridAlpha(self, binance_symbol, coinbase_symbol))
        
        self.set_portfolio_construction(EqualWeightingPortfolioConstructionModel())
        self.set_execution(ImmediateExecutionModel())
        
        # Upgrade risk management to cap losses per security
        self.set_risk_management(MaximumDrawdownPercentPerSecurity(0.05))
        
        # Use default brokerage for mixed-market compatibility
        self.set_brokerage_model(BrokerageName.DEFAULT)
        
        # Set benchmark
        self.set_benchmark("BTCUSDT")
        
        # Note: Multi-exchange arbitrage (Binance + Coinbase) not supported in live trading;
        # requires separate algorithms per brokerage for live deployment.

    def on_warmup_finished(self):
        """Called when warm-up period is complete"""
        self.debug("Warm-up period finished - ready to trade")

    def on_data(self, data):
        pass


class CustomFeeModel(FeeModel):
    def __init__(self, algorithm, fee_percent):
        self.algorithm = algorithm
        self.fee_percent = fee_percent
    
    def get_order_fee(self, parameters):
        # Apply percentage-based fee to order value
        order_value = parameters.order.absolute_quantity * parameters.order.price
        return OrderFee(CashAmount(order_value * self.fee_percent, "USD"))