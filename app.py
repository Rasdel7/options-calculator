import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Options Calculator",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Options Strategy Calculator")
st.markdown("Black-Scholes pricing, payoff diagrams, "
            "Greeks and multi-leg strategy analysis.")
st.markdown("---")

# ── Black-Scholes ─────────────────────────────
def black_scholes(S, K, T, r, sigma,
                   option_type='call'):
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    d1 = (np.log(S / K) +
          (r + 0.5 * sigma**2) * T) / \
         (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = (S * norm.cdf(d1) -
                 K * np.exp(-r * T) *
                 norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) *
                 norm.cdf(-d2) -
                 S * norm.cdf(-d1))
    return max(round(price, 4), 0)

def calculate_greeks(S, K, T, r, sigma,
                      option_type='call'):
    if T <= 0:
        return {
            'delta': 0, 'gamma': 0,
            'theta': 0, 'vega': 0,
            'rho': 0}
    d1 = (np.log(S / K) +
          (r + 0.5 * sigma**2) * T) / \
         (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    n_d1 = norm.pdf(d1)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma
    gamma = n_d1 / (S * sigma *
                     np.sqrt(T))

    # Theta (per day)
    if option_type == 'call':
        theta = (-(S * n_d1 * sigma) /
                  (2 * np.sqrt(T)) -
                  r * K * np.exp(-r * T) *
                  norm.cdf(d2)) / 365
    else:
        theta = (-(S * n_d1 * sigma) /
                  (2 * np.sqrt(T)) +
                  r * K * np.exp(-r * T) *
                  norm.cdf(-d2)) / 365

    # Vega (per 1% vol change)
    vega = S * n_d1 * np.sqrt(T) / 100

    # Rho (per 1% rate change)
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * \
              norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * \
               norm.cdf(-d2) / 100

    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 6),
        'theta': round(theta, 4),
        'vega':  round(vega, 4),
        'rho':   round(rho, 4)
    }

def payoff_call(S_T, K, premium):
    return np.maximum(S_T - K, 0) - premium

def payoff_put(S_T, K, premium):
    return np.maximum(K - S_T, 0) - premium

def implied_vol(market_price, S, K,
                 T, r, option_type='call',
                 tol=1e-6, max_iter=100):
    sigma = 0.3
    for _ in range(max_iter):
        price = black_scholes(
            S, K, T, r, sigma, option_type)
        vega  = S * norm.pdf(
            (np.log(S/K) +
             (r + 0.5*sigma**2)*T) /
            (sigma*np.sqrt(T))
        ) * np.sqrt(T)
        if abs(vega) < 1e-10:
            break
        sigma -= (price - market_price) / vega
        sigma  = max(0.001, min(sigma, 10.0))
        if abs(black_scholes(
            S, K, T, r, sigma, option_type
        ) - market_price) < tol:
            break
    return round(sigma, 4)

# ── Strategy payoff functions ─────────────────
def strategy_payoff(strategy, S_range,
                     K1, K2, T, r,
                     sigma, lots=1):
    p_call_K1 = black_scholes(
        S_range[len(S_range)//2],
        K1, T, r, sigma, 'call')
    p_put_K1  = black_scholes(
        S_range[len(S_range)//2],
        K1, T, r, sigma, 'put')
    p_call_K2 = black_scholes(
        S_range[len(S_range)//2],
        K2, T, r, sigma, 'call')
    p_put_K2  = black_scholes(
        S_range[len(S_range)//2],
        K2, T, r, sigma, 'put')

    if strategy == "Long Call":
        return (payoff_call(
            S_range, K1, p_call_K1) * lots,
            "Buy 1 Call")
    elif strategy == "Long Put":
        return (payoff_put(
            S_range, K1, p_put_K1) * lots,
            "Buy 1 Put")
    elif strategy == "Short Call":
        return (-payoff_call(
            S_range, K1, p_call_K1) * lots,
            "Sell 1 Call")
    elif strategy == "Short Put":
        return (-payoff_put(
            S_range, K1, p_put_K1) * lots,
            "Sell 1 Put")
    elif strategy == "Bull Call Spread":
        return ((payoff_call(S_range, K1,
                              p_call_K1) -
                  payoff_call(S_range, K2,
                               p_call_K2)) *
                 lots,
                "Buy K1 Call + Sell K2 Call")
    elif strategy == "Bear Put Spread":
        return ((payoff_put(S_range, K2,
                             p_put_K2) -
                  payoff_put(S_range, K1,
                              p_put_K1)) *
                 lots,
                "Buy K2 Put - Sell K1 Put")
    elif strategy == "Long Straddle":
        return ((payoff_call(S_range, K1,
                              p_call_K1) +
                  payoff_put(S_range, K1,
                              p_put_K1)) *
                 lots,
                "Buy Call + Buy Put (same K)")
    elif strategy == "Short Straddle":
        return ((-payoff_call(S_range, K1,
                               p_call_K1) -
                   payoff_put(S_range, K1,
                               p_put_K1)) *
                  lots,
                 "Sell Call + Sell Put (same K)")
    elif strategy == "Long Strangle":
        return ((payoff_call(S_range, K2,
                              p_call_K2) +
                  payoff_put(S_range, K1,
                              p_put_K1)) *
                 lots,
                "Buy OTM Call + Buy OTM Put")
    elif strategy == "Bull Put Spread":
        return ((-payoff_put(S_range, K2,
                              p_put_K2) +
                   payoff_put(S_range, K1,
                               p_put_K1)) *
                  lots,
                 "Sell K2 Put + Buy K1 Put")
    elif strategy == "Covered Call":
        stock_pnl = S_range - \
            S_range[len(S_range)//2]
        return ((stock_pnl -
                  payoff_call(S_range, K1,
                               p_call_K1)) *
                 lots,
                "Long Stock + Sell Call")
    elif strategy == "Protective Put":
        stock_pnl = S_range - \
            S_range[len(S_range)//2]
        return ((stock_pnl +
                  payoff_put(S_range, K1,
                              p_put_K1)) *
                 lots,
                "Long Stock + Buy Put")
    return (np.zeros(len(S_range)), "Unknown")

# ── Tabs ──────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Option Pricer",
    "📊 Greeks Dashboard",
    "🎯 Strategy Builder",
    "🌡️ Volatility Tools",
    "📚 Learn Options"
])

# ── Tab 1: Option Pricer ──────────────────────
with tab1:
    st.markdown("### 💰 Black-Scholes Option Pricer")

    col1, col2 = st.columns(2)

    with col1:
        S = st.number_input(
            "Stock Price (S):",
            min_value=1.0,
            value=100.0, step=1.0)
        K = st.number_input(
            "Strike Price (K):",
            min_value=1.0,
            value=100.0, step=1.0)
        T_days = st.number_input(
            "Days to Expiry:",
            min_value=1, value=30, step=1)
        T = T_days / 365
        r = st.number_input(
            "Risk-free Rate (%):",
            min_value=0.0,
            value=5.0, step=0.1) / 100
        sigma = st.number_input(
            "Volatility (%):",
            min_value=1.0,
            value=20.0, step=1.0) / 100
        option_type = st.radio(
            "Option Type:",
            ["call", "put"],
            horizontal=True)

    with col2:
        price = black_scholes(
            S, K, T, r, sigma, option_type)
        greeks = calculate_greeks(
            S, K, T, r, sigma, option_type)

        # Intrinsic value
        if option_type == 'call':
            intrinsic = max(S - K, 0)
        else:
            intrinsic = max(K - S, 0)
        time_val = price - intrinsic

        # Moneyness
        if S > K * 1.02:
            moneyness = "📈 In-the-Money (ITM)"
            mon_color = "#2ecc71"
        elif S < K * 0.98:
            moneyness = "📉 Out-of-the-Money (OTM)"
            mon_color = "#e74c3c"
        else:
            moneyness = "➡️ At-the-Money (ATM)"
            mon_color = "#f39c12"

        st.markdown(
            "<div style='background:#1a1a2e;"
            "border:2px solid #30363d;"
            "border-radius:12px;"
            "padding:24px;text-align:center'>"
            "<h1 style='color:#2ecc71;"
            "margin:0'>₹" +
            str(price) + "</h1>"
            "<p style='color:#8b949e;"
            "margin:8px 0'>" +
            option_type.upper() +
            " Option Price</p>"
            "<span style='color:" +
            mon_color + "'>" +
            moneyness + "</span>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Intrinsic Value",
                  round(intrinsic, 4))
        c2.metric("Time Value",
                  round(time_val, 4))
        c3.metric("Break-even",
                  round(K + price
                        if option_type == 'call'
                        else K - price, 2))

        st.markdown("#### 🔢 Greeks")
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Δ Delta",
                  greeks['delta'])
        g2.metric("Γ Gamma",
                  greeks['gamma'])
        g3.metric("Θ Theta/day",
                  greeks['theta'])
        g4.metric("ν Vega/1%",
                  greeks['vega'])
        g5.metric("ρ Rho/1%",
                  greeks['rho'])

    # Price vs Stock Price chart
    st.markdown("---")
    st.markdown(
        "#### 📈 Option Price vs Stock Price")

    S_range = np.linspace(
        S * 0.5, S * 1.5, 200)
    prices  = [
        black_scholes(s, K, T, r,
                       sigma, option_type)
        for s in S_range
    ]
    intrinsics = [
        max(s - K, 0) if option_type == 'call'
        else max(K - s, 0)
        for s in S_range
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S_range, y=prices,
        mode='lines',
        name='Option Price (BS)',
        line=dict(color='#3498db', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=S_range, y=intrinsics,
        mode='lines',
        name='Intrinsic Value',
        line=dict(
            color='#2ecc71',
            width=1.5, dash='dash')
    ))
    fig.add_vline(
        x=S, line_dash='dash',
        line_color='#f39c12',
        annotation_text='Current S=' +
                        str(S))
    fig.add_vline(
        x=K, line_dash='dot',
        line_color='#e74c3c',
        annotation_text='Strike K=' +
                        str(K))
    fig.update_layout(
        title='Option Price vs Underlying',
        xaxis_title='Stock Price',
        yaxis_title='Option Price',
        height=400,
        template='plotly_dark',
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig,
                    use_container_width=True)

# ── Tab 2: Greeks Dashboard ───────────────────
with tab2:
    st.markdown("### 📊 Greeks Dashboard")
    st.markdown(
        "How each Greek changes as stock "
        "price varies.")

    col1, col2 = st.columns([1, 3])

    with col1:
        g_S     = st.number_input(
            "Stock Price:", 1.0, 10000.0,
            100.0, 1.0, key="g_S")
        g_K     = st.number_input(
            "Strike:", 1.0, 10000.0,
            100.0, 1.0, key="g_K")
        g_T     = st.number_input(
            "Days to Expiry:", 1, 365,
            30, 1, key="g_T") / 365
        g_r     = st.number_input(
            "Rate (%):", 0.0, 20.0,
            5.0, 0.1, key="g_r") / 100
        g_sigma = st.number_input(
            "Volatility (%):", 1.0, 100.0,
            20.0, 1.0, key="g_sigma") / 100
        g_type  = st.radio(
            "Type:", ["call", "put"],
            horizontal=True, key="g_type")

    with col2:
        S_arr = np.linspace(
            g_S * 0.5, g_S * 1.5, 150)

        deltas = [calculate_greeks(
            s, g_K, g_T, g_r,
            g_sigma, g_type)['delta']
            for s in S_arr]
        gammas = [calculate_greeks(
            s, g_K, g_T, g_r,
            g_sigma, g_type)['gamma']
            for s in S_arr]
        thetas = [calculate_greeks(
            s, g_K, g_T, g_r,
            g_sigma, g_type)['theta']
            for s in S_arr]
        vegas  = [calculate_greeks(
            s, g_K, g_T, g_r,
            g_sigma, g_type)['vega']
            for s in S_arr]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=S_arr, y=deltas,
            name='Δ Delta',
            line=dict(color='#3498db',
                      width=2)))
        fig2.add_trace(go.Scatter(
            x=S_arr,
            y=[g*100 for g in gammas],
            name='Γ Gamma ×100',
            line=dict(color='#2ecc71',
                      width=2)))
        fig2.add_trace(go.Scatter(
            x=S_arr, y=thetas,
            name='Θ Theta',
            line=dict(color='#e74c3c',
                      width=2)))
        fig2.add_trace(go.Scatter(
            x=S_arr, y=vegas,
            name='ν Vega',
            line=dict(color='#f39c12',
                      width=2)))
        fig2.add_vline(
            x=g_S, line_dash='dash',
            line_color='white',
            annotation_text='S')
        fig2.add_vline(
            x=g_K, line_dash='dot',
            line_color='#9b59b6',
            annotation_text='K')
        fig2.update_layout(
            title='All Greeks vs Stock Price',
            xaxis_title='Stock Price',
            yaxis_title='Greek Value',
            height=450,
            template='plotly_dark'
        )
        st.plotly_chart(fig2,
                        use_container_width=True)

    # Theta decay over time
    st.markdown("#### ⏱️ Theta Decay — "
                "Time Value Erosion")
    days_arr = np.linspace(0.5, 90, 100)
    tv_arr   = [
        black_scholes(g_S, g_K,
                       d/365, g_r,
                       g_sigma, g_type) -
        max(g_S - g_K, 0)
        if g_type == 'call'
        else black_scholes(g_S, g_K,
                            d/365, g_r,
                            g_sigma, g_type) -
        max(g_K - g_S, 0)
        for d in days_arr
    ]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=days_arr, y=tv_arr,
        mode='lines', fill='tozeroy',
        line=dict(color='#e74c3c', width=2),
        fillcolor='rgba(231,76,60,0.15)',
        name='Time Value'
    ))
    fig3.update_layout(
        title='Time Value Decay '
              '(Theta Effect)',
        xaxis_title='Days to Expiry',
        yaxis_title='Time Value',
        height=300,
        template='plotly_dark'
    )
    st.plotly_chart(fig3,
                    use_container_width=True)

# ── Tab 3: Strategy Builder ───────────────────
with tab3:
    st.markdown("### 🎯 Strategy Payoff Builder")

    col1, col2 = st.columns([1, 2])

    with col1:
        st_S = st.number_input(
            "Current Stock Price:",
            1.0, 10000.0, 100.0, 1.0,
            key="st_S")
        st_K1 = st.number_input(
            "Strike 1 (K1):",
            1.0, 10000.0, 95.0, 1.0)
        st_K2 = st.number_input(
            "Strike 2 (K2) — for spreads:",
            1.0, 10000.0, 105.0, 1.0)
        st_T  = st.number_input(
            "Days to Expiry:", 1, 365,
            30, 1, key="st_T") / 365
        st_r  = st.number_input(
            "Rate (%):", 0.0, 20.0,
            5.0, 0.1, key="st_r") / 100
        st_sigma = st.number_input(
            "Volatility (%):", 1.0, 100.0,
            20.0, 1.0, key="st_sigma") / 100
        lots = st.number_input(
            "Lots (contract multiplier):",
            1, 100, 1)

        strategy = st.selectbox(
            "Strategy:",
            ["Long Call", "Long Put",
             "Short Call", "Short Put",
             "Bull Call Spread",
             "Bear Put Spread",
             "Bull Put Spread",
             "Long Straddle",
             "Short Straddle",
             "Long Strangle",
             "Covered Call",
             "Protective Put"])

    with col2:
        S_range = np.linspace(
            st_S * 0.5, st_S * 1.5, 300)

        payoffs, description = \
            strategy_payoff(
                strategy, S_range,
                st_K1, st_K2, st_T,
                st_r, st_sigma, lots)

        # Max profit/loss
        max_profit = payoffs.max()
        max_loss   = payoffs.min()
        # Breakeven(s)
        sign_changes = np.where(
            np.diff(np.sign(payoffs)))[0]
        breakevens = [
            round(S_range[i], 2)
            for i in sign_changes
        ]

        c1, c2, c3 = st.columns(3)
        c1.metric("Max Profit",
                  "₹" + str(round(
                      max_profit, 2))
                  if max_profit < 1e6
                  else "Unlimited")
        c2.metric("Max Loss",
                  "₹" + str(round(
                      max_loss, 2)))
        c3.metric("Breakeven(s)",
                  str(breakevens)
                  if breakevens
                  else "N/A")

        fig4 = go.Figure()
        colors = np.where(
            payoffs >= 0,
            'rgba(46,204,113,0.7)',
            'rgba(231,76,60,0.7)')

        fig4.add_trace(go.Scatter(
            x=S_range,
            y=payoffs,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#3498db',
                      width=2.5),
            fillcolor='rgba(52,152,219,0.1)',
            name='P&L'
        ))
        fig4.add_hline(
            y=0, line_color='white',
            line_width=1)
        fig4.add_vline(
            x=st_S, line_dash='dash',
            line_color='#f39c12',
            annotation_text='Current S')
        fig4.add_vline(
            x=st_K1, line_dash='dot',
            line_color='#2ecc71',
            annotation_text='K1')
        if st_K2 != st_K1:
            fig4.add_vline(
                x=st_K2, line_dash='dot',
                line_color='#e74c3c',
                annotation_text='K2')
        for be in breakevens:
            fig4.add_vline(
                x=be,
                line_dash='dash',
                line_color='#9b59b6',
                line_width=1)

        fig4.update_layout(
            title=strategy +
                  " — Payoff at Expiry",
            xaxis_title='Stock Price at Expiry',
            yaxis_title='Profit / Loss',
            height=450,
            template='plotly_dark'
        )
        st.plotly_chart(fig4,
                        use_container_width=True)
        st.caption("📋 " + description)

    # Strategy guide table
    st.markdown("---")
    st.markdown("#### 📋 Strategy Quick Guide")
    guide_df = pd.DataFrame({
        'Strategy': [
            'Long Call', 'Long Put',
            'Short Call', 'Short Put',
            'Bull Call Spread',
            'Bear Put Spread',
            'Long Straddle',
            'Short Straddle',
            'Covered Call'],
        'View': [
            'Bullish', 'Bearish',
            'Neutral-Bearish',
            'Neutral-Bullish',
            'Moderately Bullish',
            'Moderately Bearish',
            'High Volatility',
            'Low Volatility',
            'Neutral-Bullish'],
        'Max Profit': [
            'Unlimited', 'K-Premium',
            'Premium', 'Premium',
            'K2-K1-Net Debit',
            'K2-K1-Net Debit',
            'Unlimited', 'Net Premium',
            'Strike-Stock+Premium'],
        'Max Loss': [
            'Premium', 'Premium',
            'Unlimited', 'Unlimited',
            'Net Debit', 'Net Debit',
            'Net Debit', 'Unlimited',
            'Stock-Premium']
    })
    st.dataframe(guide_df,
                 use_container_width=True,
                 hide_index=True)

# ── Tab 4: Volatility ─────────────────────────
with tab4:
    st.markdown("### 🌡️ Volatility Tools")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔍 Implied Volatility")
        iv_S    = st.number_input(
            "Stock Price:", 1.0, 10000.0,
            100.0, 1.0, key="iv_S")
        iv_K    = st.number_input(
            "Strike:", 1.0, 10000.0,
            100.0, 1.0, key="iv_K")
        iv_T    = st.number_input(
            "Days to Expiry:", 1, 365,
            30, 1, key="iv_T") / 365
        iv_r    = st.number_input(
            "Rate (%):", 0.0, 20.0,
            5.0, 0.1, key="iv_r") / 100
        market_price = st.number_input(
            "Market Option Price:",
            0.01, 10000.0, 5.0, 0.1)
        iv_type = st.radio(
            "Option Type:",
            ["call", "put"],
            horizontal=True,
            key="iv_type")

        if st.button(
            "🔍 Calculate IV",
            type="primary"
        ):
            iv = implied_vol(
                market_price, iv_S,
                iv_K, iv_T, iv_r, iv_type)
            theoretical = black_scholes(
                iv_S, iv_K, iv_T,
                iv_r, iv, iv_type)

            st.metric(
                "Implied Volatility",
                str(round(iv * 100, 2)) + "%")
            st.metric(
                "Theoretical Price",
                round(theoretical, 4))
            st.caption(
                "IV is the market's forecast "
                "of future volatility implied "
                "by the current option price.")

    with col2:
        st.markdown("#### 📊 Vol Surface")
        st.markdown(
            "How option price changes "
            "with volatility and time.")

        base_S   = st.number_input(
            "Stock Price:", 1.0,
            10000.0, 100.0, 1.0,
            key="vs_S")
        base_K   = st.number_input(
            "Strike:", 1.0,
            10000.0, 100.0, 1.0,
            key="vs_K")
        base_r   = 0.05
        vs_type  = st.radio(
            "Option Type:",
            ["call", "put"],
            horizontal=True,
            key="vs_type")

        vol_range  = np.linspace(
            0.05, 0.80, 20)
        time_range = np.array(
            [7, 14, 30, 60, 90, 120, 180])

        Z = np.zeros(
            (len(vol_range),
             len(time_range)))
        for i, v in enumerate(vol_range):
            for j, t in enumerate(time_range):
                Z[i, j] = black_scholes(
                    base_S, base_K,
                    t/365, base_r,
                    v, vs_type)

        fig5 = go.Figure(
            data=[go.Surface(
                x=time_range,
                y=vol_range * 100,
                z=Z,
                colorscale='Viridis',
                showscale=True)])
        fig5.update_layout(
            title='Option Price Surface',
            scene=dict(
                xaxis_title='Days to Expiry',
                yaxis_title='Volatility (%)',
                zaxis_title='Option Price'),
            height=450,
            template='plotly_dark')
        st.plotly_chart(fig5,
                        use_container_width=True)

    # Volatility comparison
    st.markdown("---")
    st.markdown("#### 📈 Historical vs "
                "Implied Volatility Guide")
    vol_df = pd.DataFrame({
        'Scenario': [
            'IV > HV significantly',
            'IV ≈ HV',
            'IV < HV significantly',
            'IV rising sharply',
            'IV falling sharply'
        ],
        'Interpretation': [
            'Options expensive — consider selling',
            'Fairly priced market',
            'Options cheap — consider buying',
            'Market expects big move (fear)',
            'Calm expected (complacency)'
        ],
        'Strategy Hint': [
            'Short Straddle / Iron Condor',
            'Directional plays',
            'Long Straddle / Strangle',
            'Buy options before move',
            'Sell options for premium'
        ]
    })
    st.dataframe(vol_df,
                 use_container_width=True,
                 hide_index=True)

# ── Tab 5: Learn ──────────────────────────────
with tab5:
    st.markdown("### 📚 Options Fundamentals")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 📖 What Are Options?

        An option is a contract giving the buyer
        the **right, but not obligation**, to
        buy or sell an asset at a set price
        (strike) before/on expiry.

        **Call Option** — right to BUY
        **Put Option**  — right to SELL

        #### 💡 The 4 Basic Positions
        | Position | Pay | Max Profit | Max Loss |
        |----------|-----|-----------|---------|
        | Long Call | Premium | Unlimited | Premium |
        | Long Put  | Premium | K-Premium | Premium |
        | Short Call | Receive | Premium | Unlimited |
        | Short Put  | Receive | Premium | Unlimited |

        #### 🧮 Black-Scholes Inputs
        - **S** — Current stock price
        - **K** — Strike price
        - **T** — Time to expiry (years)
        - **r** — Risk-free interest rate
        - **σ** — Volatility (implied or historical)
        """)

    with col2:
        st.markdown("""
        #### 🔢 The Greeks Explained

        **Delta (Δ)** — How much option price
        changes per ₹1 move in stock.
        Call: 0 to 1 | Put: -1 to 0

        **Gamma (Γ)** — Rate of change of Delta.
        High near ATM, near expiry. Measures
        convexity risk.

        **Theta (Θ)** — Time decay. How much
        option loses per day. Always negative
        for buyers. Options sellers profit from
        theta.

        **Vega (ν)** — Sensitivity to volatility.
        Long options benefit from vol increase.
        Short options benefit from vol decrease.

        **Rho (ρ)** — Sensitivity to interest
        rates. Small effect for short-dated
        options.

        #### ⚠️ Key Risks
        - Options can expire worthless (100% loss)
        - Short options have unlimited loss
        - Time decay works against buyers
        - Volatility crush after earnings
        """)

    # Options vs Stocks comparison
    st.markdown("---")
    st.markdown("#### 📊 Options vs Stocks")
    compare_df = pd.DataFrame({
        'Aspect':       [
            'Max Profit', 'Max Loss',
            'Leverage', 'Time Decay',
            'Voting Rights',
            'Dividends', 'Complexity'],
        'Buying Stock': [
            'Unlimited', 'Full Investment',
            '1x', 'None',
            'Yes', 'Yes', 'Low'],
        'Buying Call':  [
            'Unlimited', 'Premium Only',
            '5-20x', 'Against You',
            'No', 'No', 'Medium'],
        'Selling Put':  [
            'Premium Only', 'Strike-Premium',
            '5-20x', 'For You',
            'No', 'No', 'Medium-High']
    })
    st.dataframe(compare_df,
                 use_container_width=True,
                 hide_index=True)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Options Strategy Calculator | "
    "Black-Scholes · Greeks · Strategies"
)