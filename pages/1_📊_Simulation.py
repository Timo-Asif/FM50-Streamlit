import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objs as go
from utils import simulate, optimal_nu

# Stock Price Process
with st.sidebar.expander("Stock Price: $dS_t = \\kappa^s \\alpha_t dt + \\sigma^s dW^s$"):
    S0 = st.number_input('Initial stock price (S0)', value=100)
    kappa_s = st.number_input('Mean reversion speed of stock price (kappa_s)', value=1)
    sigma_s = st.number_input('Volatility of stock price (sigma_s)', value=2)

# Price Signal Process
with st.sidebar.expander("Price Signal: $d\\alpha_t = \\kappa^{\\alpha} \\alpha_t dt + \\sigma^{\\alpha} dW^{\\alpha}$"):
    kappa_alpha = st.number_input('Mean reversion speed of alpha: $\\kappa^{\\alpha}$', value=10)
    sigma_alpha = st.number_input('Volatility of alpha: $\\sigma^{\\alpha}$', value=5)

# Controlled Cash Process
with st.sidebar.expander("Cash Process: $dX_t^{\\nu} = -\\nu_{t}\left(S_{t}+\kappa \\nu_{t}\\right) \mathrm{d} t $"):
    kappa = st.number_input('Execution cost parameter (kappa)', value=1e-3)

# Performance Criteria
with st.sidebar.expander("Performance Criteria: $\\phi, a$"):
    phi = st.number_input('Penalty parameter (phi)', value=0)
    a = st.number_input('Intensity of decay (a)', value=1)

# Simulation Parameters
with st.sidebar.expander("Simulation Parameters"):
    T = st.number_input('Time horizon (T)', value=1)
    num_steps = st.number_input('Number of steps (num_steps)', value=1000)
    num_simulations = st.number_input('Number of simulations (num_simulations)', value=10000)

# Run simulation with current parameters
X_T, Q_T, terminal_value, alpha, Q, X = simulate(S0, kappa_s, sigma_s, T, kappa_alpha, sigma_alpha, kappa, a, 0, phi, num_steps, num_simulations)

st.header('Simulation Results')

expected_X_T = np.mean(X_T)
expected_Q_T = np.mean(Q_T)
expected_terminal_value = np.mean(terminal_value)

std_X_T = np.std(X_T)
std_Q_T = np.std(Q_T)
std_terminal_value = np.std(terminal_value)

# Display results in three columns
st.write("### Results")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**$X_{T}^{\\nu^{*}}$**")
    st.write(f"Expected value: {expected_X_T:.2f}")
    st.write(f"Standard deviation: {std_X_T:.2f}")
with col2:
    st.markdown("**$Q_{T}^{\\nu^{*}}$**")
    st.write(f"Expected value: {expected_Q_T:.2f}")
    st.write(f"Standard deviation: {std_Q_T:.2f}")
with col3:
    st.markdown("**$X_{T}^{\\nu^{*}} + Q_{T}^{\\nu^{*}} S_{T}$**")
    st.write(f"Expected value: {expected_terminal_value:.2f}")
    st.write(f"Standard deviation: {std_terminal_value:.2f}")

# Plot histograms using Plotly
st.write("### Histograms")

fig1 = go.Figure()
fig1.add_trace(go.Histogram(x=X_T, nbinsx=50, name='$X_T^{\\nu^*}$'))
fig1.update_layout(title_text='$X_T^{\\nu^*}$ Histogram', xaxis_title='$X_T^{\\nu^*}$', yaxis_title='Frequency')

fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=Q_T, nbinsx=50, name='$Q_T^{\\nu^*}$', marker_color='green'))
fig2.update_layout(title_text='$Q_T^{\\nu^*}$ Histogram', xaxis_title='$Q_T^{\\nu^*}$', yaxis_title='Frequency')

fig3 = go.Figure()
fig3.add_trace(go.Histogram(x=terminal_value, nbinsx=50, name='$X_T^{\\nu^*} + Q_T^{\\nu^*} S_T$', marker_color='red'))
fig3.update_layout(title_text='$X_T^{\\nu^*} + Q_T^{\\nu^*} S_T$ Histogram', xaxis_title='$X_T^{\\nu^*} + Q_T^{\\nu^*} S_T$', yaxis_title='Frequency')

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)
