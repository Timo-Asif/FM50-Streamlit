import streamlit as st
import numpy as np
import plotly.graph_objs as go
from utils import simulate, optimal_nu

st.set_page_config(
    page_title="Optimal Trading Strategy Simulation",
    page_icon="📈",
)

st.write("# Optimal Trading Strategy Simulation")
st.sidebar.success("Select a task above.")

st.markdown(
    """
    This application allows you to explore different aspects of an optimal trading strategy through various tasks.
    **👈 Select a task from the sidebar** to get started!
    """
)

# Parameters
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
    num_simulations = st.number_input('Number of simulations (num_simulations)', value=3)  # Set to a small number for visualization purposes

# Run simulation with current parameters
X_T, Q_T, terminal_value, alpha, Q, X = simulate(S0, kappa_s, sigma_s, T, kappa_alpha, sigma_alpha, kappa, a, 0, phi, num_steps, num_simulations)

# Calculate control parameter nu for the sample paths
t = np.linspace(0, T, num_steps + 1)
nu = np.zeros((num_simulations, num_steps + 1))
for i in range(num_steps + 1):
    for j in range(num_simulations):
        nu[j, i] = optimal_nu(t[i], Q[j, i], alpha[j, i], a, kappa, kappa_alpha, T)

# Plot interactive graph using Plotly
st.write("## Visualization of Control Parameter $\\nu_t$")

fig = go.Figure()

for i in range(num_simulations):
    fig.add_trace(go.Scatter(
        x=t,
        y=nu[i],
        mode='lines',
        name=f'Sample Path {i+1}'
    ))

fig.update_layout(
    title="Control Parameter $$\\nu_t$$ Over Time",
    xaxis_title="Time",
    yaxis_title="$$\\nu_t$$",
    legend_title="Sample Paths",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)
