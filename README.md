# Option Pricer 

## Overview
This application implements Black-Scholes and Cox-Ross-Rubinstein binomial models for European options valuation. Compare models to leverage their distinct advantages: Black-Scholes for computational efficiency in continuous markets, binomial trees for discrete-time flexibility. Every valuation includes complete Greeks analysis (Δ, Γ, Θ, V, ρ) quantifying sensitivity to price, volatility, time decay, and interest rates. Interactive visualizations transform theoretical pricing into actionable market intelligence.

![image](https://github.com/user-attachments/assets/bc671f6f-7602-473d-963d-b3c884c3c81e)


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

## Cox-Ross-Rubenstein Binomial Tree
It is a representation of the intrinsic value an option may take at different time periods. The value of an option at any node depends on the probability that the price of the underlying asset will either increase or decrease at any given node.

### Risk Neutral Probability (p)
It represents a hypothetical probability measure under which the present value of the expected future payoff of a financial asset when discounted at risk free rate, should be equal to the current market price of the asset. It is used to discount cash flows.

## Greeks Analysis
Quantify option price sensitivities to key market parameters using the fundamental risk measures. These metrics form the foundation of risk management and hedging strategies across all pricing models. 
- Delta captures directional exposure to underlying price movements.
- Gamma measures convexity risk as Delta accelerates near-the-money.
- Theta quantifies time decay erosion. Vega assesses volatility sensitivity.
- Rho evaluates interest rate exposure.

## Website Capabilities
The web application offers a comprehensive suite of features for options analysis:

### Real-Time Functionality
- **Option Valuation**: Compute Black-Scholes or Binomial Tree based prices for any US stock option
- **Market Comparison**: Compare theoretical prices with live market quotes
- **Greeks Analysis**: Quantify option price sensitivities to key market parameters for risk management, by analyzing Delta, Gamma, Theta, Vega and Rho
- **Mispricing Detection**: Identify overvalued/undervalued options
- **Volatility Analysis**: Calculate historical volatility or input custom values

### Visual Analytics
1. **Price History**: 1-year stock price chart
2. **Volatility Analysis**: 30-day rolling volatility visualization
3. **Mispricing Heatmap**: Interactive strike-by-strike comparison
4. **Binomial Tree Visualization**: Visualizes the first 5 steps of the CRR Binomial Tree

![image](https://github.com/user-attachments/assets/b87b9d97-bafa-4366-823a-87435f38fafd) 

![image](https://github.com/user-attachments/assets/680ff1ae-0e7e-409d-864e-4edffa30de9b)

![image](https://github.com/user-attachments/assets/85638251-cb98-4896-af59-e2715959a605)

![image](https://github.com/user-attachments/assets/080f1058-b80d-41a5-b6d6-0098e52decb6)


### Technical Features
- Real-time data integration with Yahoo Finance
- Interactive parameter adjustment
- Modern UI with intuitive controls
- Responsive design for all devices

## Future Enhancements
Planned features for upcoming versions:
- **Indian Markets**: Extend model to Indian market options
- **American Options Pricing**: Implement binomial tree models for pricing American-style options
- **Portfolio Analysis**: Evaluate multiple positions simultaneously
- **Historical Backtesting**: Test strategies against historical data

## Access the Application
Live application hosted on Streamlit Sharing:  
[Interactive Options Pricer](https://options-pricer.streamlit.app/)

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
