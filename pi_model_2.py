import streamlit as st
import numpy as np
from scipy.stats import norm

st.title("Pi Trajectory Model Calculator")
st.markdown("Estimate the probability of achieving business goals under uncertainty.")

st.sidebar.header("Model Configuration")

# Mode selection
mode = st.sidebar.radio("Calculation Mode", ["Single Goal", "Composite Goals"])

# Shared inputs
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 2.0, 0.4, 0.01)
T = st.sidebar.slider("Time Remaining (Years)", 0.1, 5.0, 1.0, 0.1)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.00, step=0.01)
lambda_jump = st.sidebar.number_input("Jump Intensity (λ)", value=0.0, step=0.1)
mu_jump = st.sidebar.number_input("Average Jump Size (μ_J)", value=0.0, step=0.01)
sigma_jump = st.sidebar.number_input("Jump Std Dev (σ_J)", value=0.0, step=0.01)

# Core model logic
def single_goal_prob(F, G, sigma, T, r, lambda_jump, mu_jump, sigma_jump):
    G_discounted = G * np.exp(-r * T)
    sigma_eff_sq = sigma*2 + lambda_jump * (sigma_jump*2 + mu_jump*2)
    mu_eff = -0.5 * sigma_eff_sq + lambda_jump * mu_jump
    denominator = np.sqrt(sigma_eff_sq * T)
    if denominator == 0:
        return 0.0
    z = (np.log(G_discounted / F) + mu_eff * T) / denominator
    return norm.cdf(z)

def run_single_goal():
    F = st.number_input("Current Value (F)", min_value=1.0, value=100.0)
    G = st.number_input("Target Value (G)", min_value=1.0, value=150.0)
    prob = single_goal_prob(F, G, sigma, T, r, lambda_jump, mu_jump, sigma_jump)
    st.markdown(f"""
        *Debug Values Used*  
        F = {F}, G = {G}  
        σ = {sigma}, T = {T}, r = {r}  
        λ = {lambda_jump}, μ_J = {mu_jump}, σ_J = {sigma_jump}
        """)
    st.success(f" Probability of Success: *{prob*100:.2f}%*")

def run_composite_goals():
    st.subheader("Composite Milestones")
    num_goals = st.number_input("Number of Goals", min_value=2, max_value=10, value=2, step=1)
    goals = []
    weights = []
    use_weights = st.radio("Aggregation Method", ["Independent (all succeed)", "Weighted Average"])
    
    for i in range(num_goals):
        col1, col2, col3 = st.columns(3)
        with col1:
            F_i = st.number_input(f"F[{i+1}]", min_value=1.0, value=100.0, key=f"F_{i}")
        with col2:
            G_i = st.number_input(f"G[{i+1}]", min_value=1.0, value=150.0, key=f"G_{i}")
        with col3:
            w_i = st.number_input(f"Weight[{i+1}]", min_value=0.0, max_value=1.0, value=0.5, key=f"W_{i}")
        goals.append((F_i, G_i))
        weights.append(w_i)

    probs = [
        single_goal_prob(F_i, G_i, sigma, T, r, lambda_jump, mu_jump, sigma_jump)
        for (F_i, G_i) in goals
    ]

    if use_weights.startswith("Weighted"):
        total = np.dot(weights, probs)
        st.success(f"Weighted Probability of Success: *{total*100:.2f}%*")
    else:
        combined = np.prod(probs)
        st.success(f"Probability All Goals Succeed: *{combined*100:.2f}%*")

# Run selected mode
if mode == "Single Goal":
    run_single_goal()
else:
    run_composite_goals()