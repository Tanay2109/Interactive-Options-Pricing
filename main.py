import streamlit as st
import yfinance as yf
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import altair as alt
from scipy.stats import norm

# ─────────────── Page config ───────────────
st.set_page_config(
    page_title=" Interactive BS Pricer + Graphs",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────── Sidebar inputs ───────────────
st.sidebar.header("🛠️ Parameters")

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
      <h1 style="font-size:48px; margin-bottom: 5px;"> Black–Scholes Option Pricer</h1>
      <p style="font-size:20px; color: #f0f0f0; margin-top: 0px;">
        BS model is used for fair value pricing of European style Options. On this website you can check the fair values of a contract computed through BS method. You can also customize parameters and analyze insightful graphs! 
      </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Four top metrics: Sensex, NIFTY 50, S&P 500, Dow Jones Industrial Average
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        try:
            sensex_price_inr = yf.Ticker("^BSESN").info.get("regularMarketPrice", None)
            if sensex_price_inr is not None:
                st.metric(label="Sensex (^BSESN)", value=f"₹{sensex_price_inr:,.0f}")
        except Exception:
            st.metric(label="Sensex (^BSESN)", value="N/A")

    with col2:
        try:
            nifty_price_inr = yf.Ticker("^NSEI").info.get("regularMarketPrice", None)
            if nifty_price_inr is not None:
                st.metric(label="NIFTY 50 (^NSEI)", value=f"₹{nifty_price_inr:,.0f}")
        except Exception:
            st.metric(label="NIFTY 50 (^NSEI)", value="N/A")

    with col3:
        try:
            sp500_price_usd = yf.Ticker("^GSPC").info.get("regularMarketPrice", None)
            if sp500_price_usd is not None:
                st.metric(label="S&P 500 (^GSPC)", value=f"${sp500_price_usd:,.2f}")
        except Exception:
            st.metric(label="S&P 500 (^GSPC)", value="N/A")

    with col4:
        try:
            dow_price_usd = yf.Ticker("^DJI").info.get("regularMarketPrice", None)
            if dow_price_usd is not None:
                st.metric(label="Dow Jones (^DJI)", value=f"${dow_price_usd:,.2f}")
        except Exception:
            st.metric(label="Dow Jones (^DJI)", value="N/A")

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
        • **Mispricing Heatmap**: market vs Black–Scholes.   
        """
        )

    with intro_3:
        st.markdown("### Data Download")
        st.write(
            """
        After computation, **download** the full strike‐by‐strike mispricing CSV.  
        Use it for further offline analysis or backtesting.
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

# 4) Black–Scholes (USD only)
d1 = (np.log(S / k) + (r + 0.5 * ann_vol ** 2) * t) / (ann_vol * np.sqrt(t))
d2 = d1 - ann_vol * np.sqrt(t)
call_bs = S * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
put_bs = k * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

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
    st.subheader(" Parsed Inputs & BS Formulas")
    st.write(f"**Underlying Ticker:** {stk_name}")
    st.write(f"**Expiry Date:** {Exp_date}  ({diff_days} days to expiry)")
    st.write(f"**Strike (K):** ${k:,.2f}")
    st.write(f"**Option Type:** {'Call' if o_type == 'C' else 'Put'}")
    st.write(f"**Time to Expiry (years):** {t:.4f}")
    st.write(f"**Stock Price (S):** ${S:,.2f}")
    st.write(f"**Volatility (σ):** {ann_vol:.4f}")
    st.write(f"**Risk‐Free Rate (r):** {r:.4f}  ({r_percent:.2f}%)")

    st.markdown("---")
    st.subheader("🧮 Black–Scholes Results")
    st.write(f"• $d_1 = {d1:.4f}$")
    st.write(f"• $d_2 = {d2:.4f}$")
    st.write(f"• **Call BS Price:**  ${call_bs:,.4f}")
    st.write(f"• **Put BS Price:**   ${put_bs:,.4f}")

    st.markdown("---")
    st.subheader("Live Market Quotes (Chosen Strike)")
    if live_call is not None:
        st.write(f"• **Call Market Price:**  ${live_call:,.4f}")
        verdict_call = "Overvalued" if live_call > call_bs else "Undervalued"
        if verdict_call == "Overvalued":
            st.error(
                f"▶️ Call is OVERVALUED  (Market ${live_call:,.4f} vs BS ${call_bs:,.4f})"
            )
        else:
            st.success(
                f"✔️ Call is UNDERVALUED (Market ${live_call:,.4f} vs BS ${call_bs:,.4f})"
            )
    else:
        st.info("• Call quote not found for this strike.")

    if live_put is not None:
        st.write(f"• **Put Market Price:**   ${live_put:,.4f}")
        verdict_put = "Overvalued" if live_put > put_bs else "Undervalued"
        if verdict_put == "Overvalued":
            st.error(
                f"▶️ Put is OVERVALUED  (Market ${live_put:,.4f} vs BS ${put_bs:,.4f})"
            )
        else:
            st.success(
                f"✔️ Put is UNDERVALUED (Market ${live_put:,.4f} vs BS ${put_bs:,.4f})"
            )
    else:
        st.info("• Put quote not found for this strike.")

with col_right:
    st.markdown("## 📋 Quick Verdicts")
    if live_call is not None:
        if live_call > call_bs:
            st.warning("⚠️ Call: Overvalued")
        else:
            st.success("✅ Call: Undervalued")
    else:
        st.info("ℹ️ Call: No market quote.")

    if live_put is not None:
        if live_put > put_bs:
            st.warning("⚠️ Put: Overvalued")
        else:
            st.success("✅ Put: Undervalued")
    else:
        st.info("ℹ️ Put: No market quote.")

    st.markdown("---")
    st.markdown("## 📂 Download Mispricing Data")
    st.markdown(
        "> After the charts, a button will appear here to download the CSV."
    )
st.markdown("---")
st.header("📊 Graphical Analysis")
st.markdown(
    """
1. **Price History (1 Year)**  
2. **30-Day Rolling Volatility**  
3. **Mispricing Heatmap (Market – BS for All Strikes)**   
"""
)

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
st.subheader("3️⃣ Mispricing Heatmap (Market – BS) for All Strikes")
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

        mispricing_list.append(
            {
                "strike": strike_i,
                "OptionType": "Call",
                "Mispricing": m_call - bs_c_i,
                "MarketPrice": m_call,
                "BSPrice": bs_c_i
            }
        )
        mispricing_list.append(
            {
                "strike": strike_i,
                "OptionType": "Put",
                "Mispricing": m_put - bs_p_i,
                "MarketPrice": m_put,
                "BSPrice": bs_p_i
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
            alt.Tooltip("BSPrice:Q", title="BS Price", format="$.4f")
        ]
    ).properties(
        width=900,  # Wider chart
        height=350,  # Taller chart
        title="Market vs. Black-Scholes Mispricing"
    )

    # Add text labels to the heatmap
    text = heatmap.mark_text(
        baseline="middle",
        fontSize=9,
        fontWeight="bold",
        dx=0  # Adjust position
    ).encode(
        text=alt.Text("Mispricing:Q", format="$.2f"),
        color=alt.condition(
            "abs(datum.Mispricing) > 0.5",  # Threshold for text color
            alt.value("white"),
            alt.value("black")
        )
    )

    # Combine heatmap and text
    final_chart = (heatmap + text).configure_view(
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

# ─── CSV Download for mispricing_df ───
if "mispricing_df" in locals() and not mispricing_df.empty:
    st.markdown("---")
    st.subheader("📥 Download Mispricing Data")
    csv = mispricing_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{stk_name}_{Exp_date}_mispricing.csv",
        mime="text/csv",
    )
