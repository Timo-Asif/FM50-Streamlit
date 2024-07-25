import numpy as np
from scipy.integrate import quad
import streamlit as st

# Calculate h2(t)
def h2(t, a, kappa, T):
    return -a * kappa / (kappa + a * (T - t))

# Calculate f(t) - MODIFIED TO HANDLE SINGLE FLOAT VALUES
def f(t_val, a, kappa, kappa_alpha, T):  # Changed input to t_val (single float)
    integrand = lambda u: (kappa + a * (T - u)) / (kappa + a * (T - t_val)) * np.exp(-kappa_alpha * (u - t_val))
    integral, _ = quad(integrand, t_val, T)
    return integral  # Return the calculated integral directly

# Optimal trading strategy
def optimal_nu(t, Q, A, a, kappa, kappa_alpha, T):
    ht2 = h2(t, a, kappa, T)
    ft = f(t, a, kappa, kappa_alpha, T)  # Pass single time value to f(t)
    return (ft / (2 * kappa)) * A + (ht2 / kappa) * Q

# Simulation function
def simulate(S0, kappa_s, sigma_s, T, kappa_alpha, sigma_alpha, kappa, a, b, phi, num_steps, num_simulations):
    delta_t = T / num_steps
    t = np.linspace(0, T, num_steps + 1)

    S = np.zeros((num_simulations, num_steps + 1))
    alpha = np.zeros((num_simulations, num_steps + 1))
    Q = np.zeros((num_simulations, num_steps + 1))
    X = np.zeros((num_simulations, num_steps + 1))

    S[:, 0] = S0

    for i in range(num_steps):
        W_s = np.random.normal(0, np.sqrt(delta_t), num_simulations)
        W_alpha = np.random.normal(0, np.sqrt(delta_t), num_simulations)

        alpha[:, i + 1] = alpha[:, i] - kappa_alpha * alpha[:, i] * delta_t + sigma_alpha * W_alpha
        S[:, i + 1] = S[:, i] + alpha[:, i] * delta_t + sigma_s * W_s

        nu = optimal_nu(t[i], Q[:, i], alpha[:, i], a, kappa, kappa_alpha, T)
        Q[:, i + 1] = Q[:, i] + nu * delta_t
        X[:, i + 1] = X[:, i] - nu * (S[:, i] + kappa * nu) * delta_t

    return X[:, -1], Q[:, -1], X[:, -1] + Q[:, -1] * S[:, -1], alpha, Q, X


def get_parameters():
    st.sidebar.header('Simulation Parameters')

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

    return S0, kappa_s, sigma_s, kappa_alpha, sigma_alpha, kappa, phi, a, T, num_steps, num_simulations
