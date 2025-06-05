# Black-Scholes Option Pricer 

## Overview
This web application provides an interactive platform for pricing European options using the Black-Scholes model. It combines theoretical finance with real-time market data to deliver accurate valuations and visual analytics for options traders, students, and financial professionals.

## Black-Scholes Model
The Black-Scholes model is a mathematical framework for pricing European-style options. It calculates the theoretical value of options using five key inputs: stock price, strike price, time to expiration, risk-free rate, and volatility.

### Call Option Pricing
$$ C = S_0 N(d_1) - K e^{-rT} N(d_2) $$

### Put Option Pricing
$$ P = K e^{-rT} N(-d_2) - S_0 N(-d_1) $$

### d₁ and d₂ Parameters
The model relies on two key parameters:

$$ d_1 = \frac{\ln{\frac{S_0}{K}} + (r + \frac{\sigma^2}{2})T}{\sigma \sqrt{T}} $$

$$ d_2 = d_1 - \sigma \sqrt{T} $$

Where:
- \( C \) = Call option price
- \( P \) = Put option price
- \( S_0 \) = Current stock price
- \( K \) = Strike price
- \( T \) = Time to expiration (in years)
- \( r \) = Risk-free interest rate
- \( sigma \) = Volatility of stock returns
- \( N \) = Cumulative distribution function of standard normal distribution

## Key Assumptions
The Black-Scholes model operates under several important assumptions:
1. **European exercise**: Options can only be exercised at expiration
2. **No dividends**: The underlying stock pays no dividends
3. **Efficient markets**: No arbitrage opportunities exist
4. **Constant volatility**: Volatility remains stable during option's life
5. **Lognormal returns**: Stock prices follow geometric Brownian motion
6. **Frictionless markets**: No transaction costs or taxes
7. **Constant interest rates**: Risk-free rate remains stable

## Website Capabilities
The web application offers a comprehensive suite of features for options analysis:

### Real-Time Functionality
- **Option Valuation**: Compute Black-Scholes prices for any US stock option
- **Market Comparison**: Compare theoretical prices with live market quotes
- **Mispricing Detection**: Identify overvalued/undervalued options
- **Volatility Analysis**: Calculate historical volatility or input custom values

### Visual Analytics
1. **Price History**: 1-year stock price chart
2. **Volatility Analysis**: 30-day rolling volatility visualization
3. **Mispricing Heatmap**: Interactive strike-by-strike comparison

### Technical Features
- Real-time data integration with Yahoo Finance
- Interactive parameter adjustment
- Modern UI with intuitive controls
- Responsive design for all devices

## Future Enhancements
Planned features for upcoming versions:
- **Indian Markets**: Extend model to Indian market options
- **American Options Pricing**: Implement binomial tree models
- **Portfolio Analysis**: Evaluate multiple positions simultaneously
- **Historical Backtesting**: Test strategies against historical data

## Access the Application
Live application hosted on Streamlit Sharing:  
[Black-Scholes Option Pricer](https://tanay2109-black-scholes-option-pricing-main-kkwk9x.streamlit.app/)

## Technical Details
**Technology Stack:**
- Python 3.9+
- Streamlit (frontend framework)
- Yahoo Finance API (market data)
- Pandas (data processing)
- NumPy (mathematical operations)
- SciPy (statistical functions)
- Altair (visualizations)

## Contribute
This project welcomes contributions:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Join the discussion on improvements

*Disclaimer: This application is for educational purposes only and should not be considered financial advice. Options trading involves significant risk and may not be suitable for all investors.*
