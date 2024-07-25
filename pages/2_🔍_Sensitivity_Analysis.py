import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
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

st.header('Sensitivity Analysis Results')

t = np.linspace(0, T, num_steps + 1)
fig, ax = plt.subplots(4, 1, figsize=(12, 24))

for i in range(3):  # Plot for three sample paths
    ax[0].plot(t, alpha[i], label=f'Path {i+1}')
    ax[1].plot(t, [optimal_nu(t_val, Q[i, idx], alpha[i, idx], a, kappa, kappa_alpha, T) for idx, t_val in enumerate(t)], label=f'Path {i+1}')
    ax[2].plot(t, Q[i], label=f'Path {i+1}')
    ax[3].plot(t, X[i], label=f'Path {i+1}')

ax[0].set_title('$\\alpha_t$ over time')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('$\\alpha_t$')

ax[1].set_title('Optimal $\\nu_t^{*}$ over time')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('$\\nu_t^{*}$')

ax[2].set_title('$Q_t^{\\nu^{*}}$ over time')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('$Q_t^{\\nu^{*}}$')

ax[3].set_title('$X_t^{\\nu^{*}}$ over time')
ax[3].set_xlabel('Time')
ax[3].set_ylabel('$X_t^{\\nu^{*}}$')

for a in ax:
    a.legend()

st.pyplot(fig)
