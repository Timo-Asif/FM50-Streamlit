import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from utils import simulate

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

# Calculate stock prices (S) from the simulation function
delta_t = T / num_steps
t = np.linspace(0, T, num_steps + 1)
S = np.zeros((num_simulations, num_steps + 1))
S[:, 0] = S0
for i in range(num_steps):
    W_s = np.random.normal(0, np.sqrt(delta_t), num_simulations)
    alpha[:, i + 1] = alpha[:, i] - kappa_alpha * alpha[:, i] * delta_t + sigma_alpha * W_s
    S[:, i + 1] = S[:, i] + alpha[:, i] * delta_t + sigma_s * W_s

# Benchmark strategy
Q_benchmark = alpha
X_benchmark = np.zeros_like(Q_benchmark)
for i in range(num_steps):
    X_benchmark[:, i + 1] = X_benchmark[:, i] - Q_benchmark[:, i] * (S[:, i] + kappa * Q_benchmark[:, i]) * delta_t

P_T_optimal = X[:, -1] + Q[:, -1] * S[:, -1] - a * (Q[:, -1] ** 2) - phi * np.sum(Q ** 2, axis=1) * delta_t
P_T_benchmark = X_benchmark[:, -1] + Q_benchmark[:, -1] * S[:, -1] - a * (Q_benchmark[:, -1] ** 2) - phi * np.sum(Q_benchmark ** 2, axis=1) * delta_t

st.header('Benchmark Strategy Comparison')

fig, ax = plt.subplots(1, 2, figsize=(18, 8))

ax[0].hist(P_T_optimal, bins=50, color='blue', alpha=0.7)
ax[0].set_title('Histogram of $P_T^{\\nu^{*}}$')
ax[0].set_xlabel('$P_T^{\\nu^{*}}$')
ax[0].set_ylabel('Frequency')

ax[1].hist(P_T_benchmark, bins=50, color='orange', alpha=0.7)
ax[1].set_title('Histogram of $P_T^{\\nu^{B}}$')
ax[1].set_xlabel('$P_T^{\\nu^{B}}$')
ax[1].set_ylabel('Frequency')

st.pyplot(fig)

st.write(f"Mean $P_T^{{\\nu^{{*}}}}$: {np.mean(P_T_optimal)}")
st.write(f"Mean $P_T^{{\\nu^{{B}}}}$: {np.mean(P_T_benchmark)}")
st.write("Comparison and interpretation of the results...")
