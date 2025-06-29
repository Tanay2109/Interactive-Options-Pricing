import streamlit as st
import yfinance as yf
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ─────────────── Binomial Tree Model ───────────────
def binomial_tree_option_price(S, K, T, r, sigma, option_type, n=100):
    """
    Calculate option price using Cox-Ross-Rubinstein binomial tree model
    Parameters:
    S : current stock price
    K : strike price
    T : time to expiration (years)
    r : risk-free rate
    sigma : volatility
    option_type : 'call' or 'put'
    n : number of steps
    """
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Price tree
    price_tree = np.zeros((n + 1, n + 1))

    # Initialize asset prices at maturity
    for j in range(n + 1):
        price_tree[j, n] = S * (u ** j) * (d ** (n - j))

    # Option value at expiration
    option_tree = np.zeros((n + 1, n + 1))
    for j in range(n + 1):
        if option_type == 'call':
            option_tree[j, n] = max(0, price_tree[j, n] - K)
        else:  # put option
            option_tree[j, n] = max(0, K - price_tree[j, n])

    # Backward induction
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_tree[j, i] = np.exp(-r * dt) * (
                p * option_tree[j + 1, i + 1] + (1 - p) * option_tree[j, i + 1]
            )

    return option_tree[0, 0]

def calculate_greeks(S, K, T, r, sigma, option_type):
    """Calculate option Greeks for both call and put options"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)

    if option_type == 'call':
        delta = N_d1
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2) / 365
        rho = K * T * np.exp(-r * T) * N_d2 / 100
    else:  # put
        delta = N_d1 - 1
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = nd1 / (S * sigma * np.sqrt(T))
    vega = S * nd1 * np.sqrt(T) / 100

    return delta, gamma, theta, vega, rho


# ─────────────── Page config ───────────────
st.set_page_config(
    page_title=" Interactive Option Pricer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────── Sidebar inputs ───────────────
st.sidebar.header("🛠️ Parameters")

# Model selection
pricing_model = st.sidebar.radio(
    "Select Pricing Model:",
    ("Black-Scholes", "Binomial Tree"),
    index=0,
    help="Choose between closed-form Black-Scholes or CRR Binomial Tree method"
)

# Binomial steps slider (only shown when Binomial Tree is selected)
n_steps = 100
if pricing_model == "Binomial Tree":
    n_steps = st.sidebar.slider(
        "Number of Binomial Steps:",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="More steps = more accurate but slower computation"
    )

# 1) Option contract name
opt_name = st.sidebar.text_input(
    "Contract Name:",
    value="AAPL250619C00145000",
    help="Format: <TICKER><YYMMDD expiry><C or P><strike×1000>. e.g. AAPL250619C00145000",
)

# 2) Risk‐free rate slider (in %)
r_percent = st.sidebar.slider(
    "Risk‐Free Rate (%)", min_value=0.0, max_value=10.0, value=1.5, step=0.01
)
r = r_percent / 100.0

# 3) Volatility override
use_custom_vol = st.sidebar.checkbox(
    "Override Volatility (annualized)", value=False
)
custom_vol = None
if use_custom_vol:
    custom_vol = st.sidebar.slider(
        "σ (annualized vol)", min_value=0.01, max_value=2.00, value=0.30, step=0.001
    )


# 4) Historical window
hist_days = st.sidebar.number_input(
    "History Window (days)", min_value=30, max_value=504, value=252, step=1
)

compute_button = st.sidebar.button("Compute Option Prices")


# ─────────────── Main Content ───────────────
if not compute_button:
    # ─── Landing Page ───
    st.markdown(
        """
    <div style="text-align: center; 
                padding: 30px; 
                background-color: #004466; 
                border-radius: 10px;
                color: white;">
      <h1 style="font-size:48px; margin-bottom: 5px;"> Option Pricer </h1>
      <p style="font-size:20px; color: #f0f0f0; margin-top: 0px;">
        This application implements industry-standard Black-Scholes and Cox-Ross-Rubinstein binomial tree models to compute fair values for European-style options contracts. All valuations include comprehensive Greeks evaluation - Delta, Gamma, Theta, Vega, and Rho - quantifying risk exposures to underlying price movements, volatility shifts, time decay, and interest rate changes. While, interactive visualizations transform theoretical pricing into actionable market insights!
      </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("""
    <style>
    .blink-up {
        animation: blink-up 1s step-start infinite;
        color: green;
    }
    .blink-down {
        animation: blink-down 1s step-start infinite;
        color: red;
    }
    @keyframes blink-up {
        50% { opacity: 0.5; }
    }
    @keyframes blink-down {
        50% { opacity: 0.5; }
    }
    </style>
    """, unsafe_allow_html=True)

    def get_market_metric(ticker, currency_symbol, label):
        try:
            data = yf.Ticker(ticker).info
            current_price = data.get("regularMarketPrice", None)
            previous_close = data.get("regularMarketPreviousClose", None)

            if current_price is not None and previous_close is not None:
                change = current_price - previous_close
                change_pct = (change / previous_close) * 100

                # Create indicator
                if change >= 0:
                    indicator = f"""<span class="blink-up">▲ {abs(change):.2f} ({abs(change_pct):.2f}%)</span>"""
                else:
                    indicator = f"""<span class="blink-down">▼ {abs(change):.2f} ({abs(change_pct):.2f}%)</span>"""

                # Display metric
                st.markdown(f"""
                <div style="margin-bottom: 10px;">
                    <div style="font-size: 14px; color: #666;">{label}</div>
                    <div style="font-size: 20px; font-weight: bold;">
                        {currency_symbol}{current_price:,.2f}
                        <span style="font-size: 14px; margin-left: 5px;">{indicator}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric(label=label, value="N/A")
        except Exception:
            st.metric(label=label, value="N/A")

    # Create columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        get_market_metric("^BSESN", "₹", "Sensex")

    with col2:
        get_market_metric("^NSEI", "₹", "NIFTY 50")

    with col3:
        get_market_metric("^GSPC", "$", "S&P 500")

    with col4:
        get_market_metric("^DJI", "$", "Dow Jones")

    # ─── New Model Comparison Section ───
    st.markdown("## Model Comparison")
    comparison_table = """
| Feature               | Black-Scholes Model                          | Binomial Tree Model                          |
|-----------------------|---------------------------------------------|---------------------------------------------|
| **Strengths**                                                                                                   |
|                       | • Closed-form solution                      | • Handles American/exotic options           |
|                       | • Industry standard for European options    | • Intuitive price evolution visualization   |
|                       | • Computationally efficient                 | • Flexible for dividends/early exercise     |
| **Weaknesses**                                                                                                  |
|                       | • Only prices European options              | • Computationally intensive                |
|                       | • Constant volatility assumption            | • Accuracy depends on number of steps       |
|                       | • Assumes no dividends                      | • Slower for deep out-of-money options      |
| **Key Assumptions**                                                                                             |
|                       | 1. Log-normal stock returns                 | 1. Discrete time steps                     |
|                       | 2. No dividends                            | 2. Constant up/down factors                |
|                       | 3. No transaction costs                    | 3. Risk-neutral probability                |
|                       | 4. Constant Volatility                      | 4. Complete markets                        |
|                       | 5. Constant risk-free rate                 | 5. No arbitrage opportunities              |
"""

    st.markdown(comparison_table)

    st.markdown("""
**Note:** The Binomial Tree converges to Black-Scholes prices as the number of steps tend to infinity(n → ∞).
""")

    # Add some visual separation
    st.markdown("---")

    # Introduction columns
    intro_1, intro_2, intro_3 = st.columns(3)
    with intro_1:
        st.markdown("### How It Works")
        st.write(
            """
        1. **Enter** an option contract name (e.g. `AAPL250619C00145000`).  
        2. **Adjust** risk‐free rate & volatility (or let it auto‐compute).  
        3. **Click** “Compute Option Prices.”  
        """
        )

    with intro_2:
        st.markdown("### 📊 Visual Analysis")
        st.write(
            """
        • **Price History** over the past year.  
        • **30‐Day Rolling Volatility**.  
        • **Mispricing Heatmap**: market vs model price.
        • **Binomial Tree Visualization**.   
        """
        )

    with intro_3:
        st.markdown("### Greeks Analysis")
        st.write(
            """
        Quantify option price sensitivities to key market parameters for risk management by analyzing Greeks (Delta, Gamma, Theta, Vega and Rho).
        """
        )

    st.markdown("---")
    st.write(
        """
    Ready? Fill in the parameters on the left and hit **Compute Option Prices**!
    """
    )
    st.stop()

# ─── “Compute Option Prices” block ───

# 1) Parse option contract name using robust method
def parse_option_contract(contract):
    # Find the first digit (start of expiry date)
    first_digit_idx = next((i for i, c in enumerate(contract) if c.isdigit()), None)
    if first_digit_idx is None:
        raise ValueError("No expiry date found in contract name")

    # Extract ticker symbol (everything before first digit)
    ticker = contract[:first_digit_idx]

    # Extract expiry date (6 digits after ticker)
    if len(contract) < first_digit_idx + 6:
        raise ValueError("Contract name too short for expiry date")
    expiry_str = contract[first_digit_idx:first_digit_idx+6]
    try:
        expiry_date = datetime.strptime(expiry_str, "%y%m%d").date()
    except ValueError:
        raise ValueError(f"Invalid expiry date format: {expiry_str}")

    # Extract option type (single character after expiry date)
    if len(contract) < first_digit_idx + 7:
        raise ValueError("Contract name too short for option type")
    option_type = contract[first_digit_idx+6].upper()
    if option_type not in ('C', 'P'):
        raise ValueError(f"Invalid option type: {option_type}")

    # Extract strike price (remaining characters)
    strike_str = contract[first_digit_idx+7:]
    if not strike_str.isdigit():
        raise ValueError(f"Invalid strike price: {strike_str}")

    # Convert strike price (stored as integer * 1000)
    strike = int(strike_str) / 1000.0

    return ticker, expiry_date, option_type, strike

try:
    stk_name, Exp_date, o_type, k = parse_option_contract(opt_name)

    Curr_date = date.today()
    diff_days = (Exp_date - Curr_date).days
    if diff_days < 0:
        raise ValueError(f"Expiry {Exp_date} is in the past.")
    t = diff_days / 365.0

except Exception as e:
    st.sidebar.error(f"Error parsing contract name: {e}")
    st.stop()

# 2) Fetch underlying price + 1-year history
try:
    ticker = yf.Ticker(stk_name)
    S = ticker.info.get("regularMarketPrice", None)
    if S is None:
        raise ValueError("No market price available for ticker.")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    hist = yf.download(stk_name, start=start_date, end=end_date, progress=False)
    if hist.shape[0] < 2:
        raise ValueError("Not enough history to compute volatility/graphs.")
except Exception as e:
    st.sidebar.error(f"Error fetching data for {stk_name}: {e}")
    st.stop()

# 3) Volatility (annualized)
if use_custom_vol:
    ann_vol = float(custom_vol)
else:
    daily_ret = hist["Close"].pct_change().dropna()
    daily_vol = float(daily_ret.std())
    ann_vol = daily_vol * np.sqrt(252)

if ann_vol <= 0 or t <= 0:
    st.sidebar.error("Volatility and time‐to‐expiry must be > 0.")
    st.stop()

# ─── Display results ───
col_left, col_right = st.columns((2, 1))

# 5) Live option quotes for that strike
try:
    opt_chain = ticker.option_chain()
    calls_df = opt_chain.calls
    puts_df = opt_chain.puts

    call_row = calls_df[calls_df["strike"] == k]
    put_row = puts_df[puts_df["strike"] == k]

    live_call = float(call_row["lastPrice"]) if not call_row.empty else None
    live_put = float(put_row["lastPrice"]) if not put_row.empty else None
except Exception:
    live_call, live_put = None, None

# ─── Two‐column layout for results ───
col_left, col_right = st.columns((2, 1))

with col_left:
    st.subheader(" Parsed Inputs & Formulas")
    st.write(f"**Underlying Ticker:** {stk_name}")
    st.write(f"**Expiry Date:** {Exp_date}  ({diff_days} days to expiry)")
    st.write(f"**Strike (K):** ${k:,.2f}")
    st.write(f"**Option Type:** {'Call' if o_type == 'C' else 'Put'}")
    st.write(f"**Time to Expiry (years):** {t:.4f}")
    st.write(f"**Stock Price (S):** ${S:,.2f}")
    st.write(f"**Volatility (σ):** {ann_vol:.4f}")
    st.write(f"**Risk‐Free Rate (r):** {r:.4f}  ({r_percent:.2f}%)")
    st.markdown("---")

# ─── Calculate option prices based on selected model ───
if pricing_model == "Black-Scholes":
            # Black-Scholes calculation (existing code)
            d1 = (np.log(S / k) + (r + 0.5 * ann_vol ** 2) * t) / (ann_vol * np.sqrt(t))
            d2 = d1 - ann_vol * np.sqrt(t)
            call_price = S * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
            put_price = k * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

            model_details = f"""
            **Black-Scholes Model:**
            - $d_1 = {d1:.4f}$
            - $d_2 = {d2:.4f}$
            - Call Price: ${call_price:.4f}
            - Put Price: ${put_price:.4f}
            """
else:
    # Binomial Tree calculation
    with st.spinner(f"Calculating Binomial Tree with {n_steps} steps..."):
        call_price = binomial_tree_option_price(S, k, t, r, ann_vol, 'call', n_steps)
        put_price = binomial_tree_option_price(S, k, t, r, ann_vol, 'put', n_steps)

        model_details = f"""
                **Binomial Tree Model:**
                - Steps: {n_steps}
                - Call Price: ${call_price:.4f}
                - Put Price: ${put_price:.4f}
                """
st.subheader(f" {pricing_model} Results")
st.markdown(model_details)

st.markdown("---")
st.subheader("Live Market Quotes (Chosen Strike)")
if live_call is not None:
        st.write(f"• **Call Market Price:**  ${live_call:,.4f}")
        verdict_call = "Overvalued" if live_call > call_price else "Undervalued"
        if verdict_call == "Overvalued":
            st.error(
                f"▶️ Call is OVERVALUED  (Market ${live_call:,.4f}  vs  {pricing_model} ${call_price:,.4f})"
            )
        else:
            st.success(
                f"✔️ Call is UNDERVALUED (Market ${live_call:,.4f}  vs  {pricing_model} ${call_price:,.4f})"
            )
else:
        st.info("• Call quote not found for this strike.")

if live_put is not None:
        st.write(f"• **Put Market Price:**   ${live_put:,.4f}")
        verdict_put = "Overvalued" if live_put > put_price else "Undervalued"
        if verdict_put == "Overvalued":
            st.error(
                f"▶️ Put is OVERVALUED  (Market ${live_put:,.4f}  vs  {pricing_model}  ${put_price:,.4f})"
            )
        else:
            st.success(
                f"✔️ Put is UNDERVALUED (Market ${live_put:,.4f}  vs  {pricing_model}  ${put_price:,.4f})"
            )
else:
        st.info("• Put quote not found for this strike.")

with col_right:
    st.markdown("## 📋 Quick Verdicts")
    if live_call is not None:
        if live_call > call_price:
            st.warning("⚠️ Call: Overvalued")
        else:
            st.success("✅ Call: Undervalued")
    else:
        st.info("ℹ️ Call: No market quote.")

    if live_put is not None:
        if live_put > put_price:
            st.warning("⚠️ Put: Overvalued")
        else:
            st.success("✅ Put: Undervalued")
    else:
        st.info("ℹ️ Put: No market quote.")


# Greeks Analysis
st.markdown("---")
st.subheader("Greeks Analysis")

try:
    # Calculate Greeks
    call_delta, call_gamma, call_theta, call_vega, call_rho = calculate_greeks(
        S, k, t, r, ann_vol, 'call'
    )
    put_delta, put_gamma, put_theta, put_vega, put_rho = calculate_greeks(
        S, k, t, r, ann_vol, 'put'
    )

    # Create and display DataFrame
    greeks_data = {
        'Greek': ['Delta', 'Gamma', 'Theta (daily)', 'Vega (per 1%)', 'Rho (per 1%)'],
        'Call': [
            f"{call_delta:.4f}",
            f"{call_gamma:.6f}",
            f"{call_theta:.6f}",
            f"{call_vega:.4f}",
            f"{call_rho:.4f}"
        ],
        'Put': [
            f"{put_delta:.4f}",
            f"{put_gamma:.6f}",
            f"{put_theta:.6f}",
            f"{put_vega:.4f}",
            f"{put_rho:.4f}"
        ]
    }
    greeks_df = pd.DataFrame(greeks_data)

    # Display styled table - fixed Styler error
    st.dataframe(
        greeks_df.style
        .format(precision=4)
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([{
            'selector': 'th',
            'props': [('text-align', 'center'), ('background-color', '#0E1117'), ('color', 'white')]
        }])
    )

    # Create tabs for each Greek
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Delta", "Gamma", "Theta", "Vega", "Rho"])
    with tab1:  # Delta
        st.markdown("**Delta Sensitivity**")
        st.caption("Measures rate of change of option value with respect to underlying price")

        # Create Delta visualization
        delta_df = pd.DataFrame({
            'Option': ['Call', 'Put'],
            'Value': [call_delta, put_delta]
        })
        delta_chart = alt.Chart(delta_df).mark_bar().encode(
            x='Option',
            y='Value',
            color=alt.condition(
                alt.datum.Value > 0,
                alt.value('#4CAF50'),  # Green
                alt.value('#F44336')   # Red
            )
        ).properties(width=400, height=300, title="Delta Values")
        st.altair_chart(delta_chart, use_container_width=True)

        # Show metrics below chart
        st.metric("Call Delta", f"{call_delta:.4f}")
        st.metric("Put Delta", f"{put_delta:.4f}")

    with tab2:  # Gamma
        st.markdown("**Gamma Sensitivity**")
        st.caption("Measures rate of change of Delta with respect to underlying price")

        gamma_df = pd.DataFrame({
            'Option': ['Call', 'Put'],
            'Value': [call_gamma, put_gamma]
        })
        gamma_chart = alt.Chart(gamma_df).mark_bar().encode(
            x='Option',
            y='Value',
            color=alt.value('#2196F3')  # Blue for positive-only Gamma
        ).properties(width=400, height=300, title="Gamma Values")
        st.altair_chart(gamma_chart, use_container_width=True)

        st.metric("Call Gamma", f"{call_gamma:.6f}")
        st.metric("Put Gamma", f"{put_gamma:.6f}")

    with tab3:  # Theta
        st.markdown("**Theta (Time Decay)**")
        st.caption("Measures sensitivity to time passing (daily decay)")

        theta_df = pd.DataFrame({
            'Option': ['Call', 'Put'],
            'Value': [call_theta, put_theta]
        })
        theta_chart = alt.Chart(theta_df).mark_bar().encode(
            x='Option',
            y='Value',
            color=alt.condition(
                alt.datum.Value > 0,
                alt.value('#4CAF50'),  # Green
                alt.value('#F44336')   # Red
            )
        ).properties(width=400, height=300, title="Theta Values (Daily)")
        st.altair_chart(theta_chart, use_container_width=True)

        st.metric("Call Theta", f"{call_theta:.6f}")
        st.metric("Put Theta", f"{put_theta:.6f}")

    with tab4:  # Vega
        st.markdown("**Vega (Volatility Sensitivity)**")
        st.caption("Measures sensitivity to 1% change in implied volatility")

        vega_df = pd.DataFrame({
            'Option': ['Call', 'Put'],
            'Value': [call_vega, put_vega]
        })
        vega_chart = alt.Chart(vega_df).mark_bar().encode(
            x='Option',
            y='Value',
            color=alt.value('#9C27B0')  # Purple for positive-only Vega
        ).properties(width=400, height=300, title="Vega Values (per 1% change)")
        st.altair_chart(vega_chart, use_container_width=True)

        st.metric("Call Vega", f"{call_vega:.4f}")
        st.metric("Put Vega", f"{put_vega:.4f}")

    with tab5:  # Rho
        st.markdown("**Rho (Interest Rate Sensitivity)**")
        st.caption("Measures sensitivity to 1% change in risk-free rate")

        rho_df = pd.DataFrame({
            'Option': ['Call', 'Put'],
            'Value': [call_rho, put_rho]
        })
        rho_chart = alt.Chart(rho_df).mark_bar().encode(
            x='Option',
            y='Value',
            color=alt.condition(
                alt.datum.Value > 0,
                alt.value('#4CAF50'),  # Green
                alt.value('#F44336')   # Red
            )
        ).properties(width=400, height=300, title="Rho Values (per 1% change)")
        st.altair_chart(rho_chart, use_container_width=True)

        st.metric("Call Rho", f"{call_rho:.4f}")
        st.metric("Put Rho", f"{put_rho:.4f}")
except Exception as e:
    st.error(f"Error calculating Greeks: {str(e)}")
    st.warning("Please check your input parameters and try again")

st.markdown("---")
st.header("Graphical Analysis")

# ─── 1) Price History ───
st.subheader("1️⃣ Price History (Last 1 Year)")

# Create the DataFrame properly by converting to Series
closing_prices = hist["Close"].squeeze()  # Convert to 1D Series
price_df = pd.DataFrame({"Closing Price": closing_prices})

# Format the chart
st.line_chart(price_df)

# ─── 2) 30-Day Rolling Volatility ───
st.subheader("2️⃣ 30-Day Rolling Volatility")

# Calculate the rolling volatility and flatten to 1D if needed
rolling_series = (
    hist["Close"].pct_change().rolling(30).std().dropna() * np.sqrt(252)
)

# Ensure it's a Series, not a DataFrame or 2D array
if isinstance(rolling_series, pd.DataFrame):
    rolling_series = rolling_series.squeeze()

# Create DataFrame properly
vol_df = pd.DataFrame(
    {"30d Rolling Volatility": rolling_series}, index=rolling_series.index
)

# Plot
st.line_chart(vol_df)

# ─── 3) Mispricing Heatmap ───
if pricing_model=="Black-Scholes":
        st.subheader("3️⃣ Mispricing Heatmap (Market vs BS) for All Strikes")
        try:
            opt_chain_full = ticker.option_chain()
            calls_full = opt_chain_full.calls
            puts_full = opt_chain_full.puts

            common_strikes = sorted(
                set(calls_full["strike"]).intersection(puts_full["strike"])
            )

            mispricing_list = []
            for strike_i in common_strikes:
                row_c = calls_full[calls_full["strike"] == strike_i]
                row_p = puts_full[puts_full["strike"] == strike_i]
                m_call = float(row_c["lastPrice"]) if not row_c.empty else np.nan
                m_put = float(row_p["lastPrice"]) if not row_p.empty else np.nan

                # Calculate BS prices in USD
                d1_i = (np.log(S / strike_i) + (r + 0.5 * ann_vol ** 2) * t) / (
                    ann_vol * np.sqrt(t)
                )
                d2_i = d1_i - ann_vol * np.sqrt(t)
                bs_c_i = S * norm.cdf(d1_i) - strike_i * np.exp(-r * t) * norm.cdf(d2_i)
                bs_p_i = strike_i * np.exp(-r * t) * norm.cdf(-d2_i) - S * norm.cdf(-d1_i)

                call_delta, call_gamma, call_theta, call_vega, call_rho = calculate_greeks(
                S, strike_i, t, r, ann_vol, 'call')
                put_delta, put_gamma, put_theta, put_vega, put_rho = calculate_greeks(
                S, strike_i, t, r, ann_vol, 'put')
                mispricing_list.append(
                    {
                        "strike": strike_i,
                        "OptionType": "Call",
                        "Mispricing": m_call - bs_c_i,
                        "MarketPrice": m_call,
                        "BSPrice": bs_c_i,
                        "Delta": call_delta,
                        "Gamma": call_gamma,
                        "Theta": call_theta,
                        "Vega": call_vega,
                        "Rho": call_rho
                    }
                )
                mispricing_list.append(
                    {
                        "strike": strike_i,
                        "OptionType": "Put",
                        "Mispricing": m_put - bs_p_i,
                        "MarketPrice": m_put,
                        "BSPrice": bs_p_i,
                        "Delta": put_delta,
                        "Gamma": put_gamma,
                        "Theta": put_theta,
                        "Vega": put_vega,
                        "Rho": put_rho
                    }
                )

            mispricing_df = pd.DataFrame(mispricing_list)

            # Filter out strikes with no pricing data
            mispricing_df = mispricing_df.dropna(subset=["Mispricing"])

            # Create the heatmap
            heatmap = alt.Chart(mispricing_df).mark_rect().encode(
                x=alt.X(
                    "strike:O",
                    sort=list(sorted(mispricing_df["strike"].unique())),
                    axis=alt.Axis(
                        labelAngle=-45,
                        title="Strike Price ($)",
                        labelFontSize=10,
                        titleFontSize=12
                    ),
                    scale=alt.Scale(padding=0.5)  # Add padding between columns
                ),
                y=alt.Y(
                    "OptionType:N",
                    title="Option Type",
                    axis=alt.Axis(
                        titleFontSize=12,
                        labelFontSize=11
                    )
                ),
                color=alt.Color(
                    "Mispricing:Q",
                    scale=alt.Scale(
                        scheme="redblue",
                        domainMid=0,  # Center at zero for better divergence
                        reverse=False  # Red for positive, blue for negative
                    ),
                    legend=alt.Legend(
                        title="Mispricing ($)",
                        titleFontSize=11,
                        labelFontSize=10,
                        gradientLength=300
                    )
                ),
                tooltip=[
                    alt.Tooltip("strike:Q", title="Strike Price", format="$.2f"),
                    alt.Tooltip("OptionType:N", title="Option Type"),
                    alt.Tooltip("Mispricing:Q", title="Mispricing", format="$.4f"),
                    alt.Tooltip("MarketPrice:Q", title="Market Price", format="$.4f"),
                    alt.Tooltip("TheoreticalPrice:Q", title=f"{pricing_model} Price", format="$.4f"),
                    alt.Tooltip("Delta:Q", title="Delta", format=".4f"),
                    alt.Tooltip("Gamma:Q", title="Gamma", format=".6f"),
                    alt.Tooltip("Theta:Q", title="Theta", format=".6f"),
                    alt.Tooltip("Vega:Q", title="Vega", format=".4f"),
                    alt.Tooltip("Rho:Q", title="Rho", format=".4f")
                ]
            ).properties(
                width=1200,  # Wider chart
                height=350,  # Taller chart

            )

            # Add text labels to the heatmap
            #text = heatmap.mark_text(
             #   baseline="middle",
              #  fontSize=9,
               # fontWeight="bold",
                #dx=0  # Adjust position
            #).encode(
             #   text=alt.Text("Mispricing:Q", format="$.2f"),
              ##  color=alt.condition(
                #    "abs(datum.Mispricing) > 0.5",  # Threshold for text color
                 #   alt.value("white"),
                  #  alt.value("black")
                #)
            #)

            # Combine heatmap and text
            final_chart = (heatmap).configure_view(
                strokeWidth=0  # Remove border
            ).configure_axis(
                grid=False  # Remove grid lines
            ).configure_title(
                fontSize=16,
                anchor="middle"
            ).interactive()  # Enable zoom/pan

            st.altair_chart(final_chart, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not build mispricing heatmap: {e}")
else:

        def plot_binomial_tree(S, K, T, r, sigma, option_type, n_steps=5):
            """Enhanced binomial tree visualization with dark theme"""
            plt.style.use('dark_background')

            # Create figure with dark background
            fig, (ax_stock, ax_option) = plt.subplots(1, 2, figsize=(18, 10),
                                            facecolor='#0E1117', gridspec_kw={'wspace': 0.15})
            fig.suptitle(f'Binomial Tree: {option_type.capitalize()} Option (First {n_steps} Steps)',
                        color='white', fontsize=18, y=1.02)

            # Calculate tree parameters
            dt = T / n_steps
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(r * dt) - d) / (u - d)

            # Initialize trees
            stock_tree = np.zeros((n_steps + 1, n_steps + 1))
            option_tree = np.zeros((n_steps + 1, n_steps + 1))
            stock_tree[0, 0] = S

            # Build stock price tree
            for j in range(1, n_steps + 1):
                stock_tree[0, j] = stock_tree[0, j-1] * u
                for i in range(1, j + 1):
                    stock_tree[i, j] = stock_tree[i-1, j-1] * d

            # Calculate option values
            for i in range(n_steps + 1):
                if option_type == 'call':
                    option_tree[i, n_steps] = max(0, stock_tree[i, n_steps] - K)
                else:  # put option
                    option_tree[i, n_steps] = max(0, K - stock_tree[i, n_steps])

            # Backward induction
            for j in range(n_steps - 1, -1, -1):
                for i in range(j + 1):
                    option_tree[i, j] = np.exp(-r * dt) * (
                        p * option_tree[i, j+1] + (1-p) * option_tree[i+1, j+1])

            # Enhanced styling parameters
            node_style = {
                'ha': 'center',
                'va': 'center',
                'fontsize': 12,  # Increased font size
                'fontweight': 'bold',
                'bbox': dict(
                    boxstyle="round,pad=0.5",  # More padding
                    ec='none',
                    lw=2,
                    alpha=0.95
                )
            }

            line_style = {
                'color': '#4FB99F',
                'lw': 2.5,  # Thicker lines
                'alpha': 0.8,
                'solid_capstyle': 'round'
            }

            # Node size and spacing parameters
            node_width = 1.5
            node_height = 1.0
            x_spacing = 4
            y_spacing = 2.5

            # Plot stock price tree
            for j in range(n_steps + 1):
                for i in range(j + 1):
                    y_pos = (j - i) * y_spacing - (j * y_spacing/2)

                    # Stock price nodes (larger and more visible)
                    ax_stock.add_patch(Rectangle(
                        (j * x_spacing - node_width/2, y_pos - node_height/2),
                        node_width, node_height,
                        facecolor='#1F4E79',  # Darker blue for contrast
                        edgecolor='#F2AA4C',
                        lw=1.5
                    ))
                    ax_stock.text(
                        j * x_spacing, y_pos,
                        f"{stock_tree[i, j]:.2f}",
                        color='#000000',  # Black text
                        **node_style
                    )

                    # Option value nodes (larger and more visible)
                    opt_color = '#2E8B57' if option_tree[i, j] > 0 else '#B22222'  # Darker green/red
                    ax_option.add_patch(Rectangle(
                        (j * x_spacing - node_width/2, y_pos - node_height/2),
                        node_width, node_height,
                        facecolor=opt_color,
                        edgecolor='white',
                        lw=1.5
                    ))
                    ax_option.text(
                        j * x_spacing, y_pos,
                        f"{option_tree[i, j]:.2f}",
                        color='#000000',
                        **node_style
                    )

                    # Connections (thicker and more visible)
                    if j < n_steps:
                        for ax in [ax_stock, ax_option]:
                            ax.plot(
                                [j * x_spacing, (j+1) * x_spacing],
                                [y_pos, y_pos + y_spacing/2],
                                **line_style
                            )
                            ax.plot(
                                [j * x_spacing, (j+1) * x_spacing],
                                [y_pos, y_pos - y_spacing/2],
                                **line_style
                            )

            # Formatting adjustments
            for ax, title in zip([ax_stock, ax_option],
                                ["Stock Price ($)", f"Option Value ($)"]):
                ax.set_title(title, color='white', pad=25, fontsize=16)
                ax.axis('off')
                ax.set_xlim(-1, n_steps * x_spacing + 1)
                ax.set_ylim(-n_steps * y_spacing/2 - 2, n_steps * y_spacing/2 + 2)

            plt.tight_layout()
            return fig

        # ─── In your Streamlit visualization section ───
        if pricing_model == "Binomial Tree":
            st.subheader("3️⃣  Binomial Tree Visualization")

            # Create tabs for Call/Put visualization
            tab1, tab2 = st.tabs(["Call Option Tree", "Put Option Tree"])

            with tab1:
                with st.spinner("Generating Call Option Tree..."):
                    fig_call = plot_binomial_tree(S, k, t, r, ann_vol, 'call', min(n_steps, 5))
                    st.pyplot(fig_call)

            with tab2:
                with st.spinner("Generating Put Option Tree..."):
                    fig_put = plot_binomial_tree(S, k, t, r, ann_vol, 'put', min(n_steps, 5))
                    st.pyplot(fig_put)

            st.markdown(f"""
            <style>
            .stTabs [data-baseweb="tab-list"] {{
                gap: 10px;
            }}
            .stTabs [data-baseweb="tab"] {{
                padding: 8px 16px;
                background: #0E1117;
                border-radius: 4px 4px 0 0;
            }}
            </style>
            
            **Visualization Features:**
            - ⚫ **Stock Prices and Option values** shown in Black
            - 🟢 **In-the-Money** nodes in mint green
            - 🔴 **Out-of-the-Money** nodes in coral red<br/>
            *Note: Displaying first 5 steps (full calculation uses {n_steps} steps)*
            """, unsafe_allow_html=True)
