import pandas as pd
import streamlit as st
from env import NetworkEnv
from dqn_agent import DQNAgent
from train import (
    train_dqn,
    run_random_policy,
    run_static_policy,
    run_best_latency_policy,
    evaluate_dqn
)

st.set_page_config(
    page_title="Network Traffic Load Balancer",
    page_icon="🌐",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 5px;
}
.sub-title {
    font-size: 18px;
    color: #9aa0a6;
    margin-bottom: 25px;
}
.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 10px;
}
.card {
    padding: 18px;
    border-radius: 16px;
    background-color: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 15px;
}
.best-box {
    padding: 18px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(0,128,255,0.18), rgba(0,255,170,0.12));
    border: 1px solid rgba(255,255,255,0.12);
    font-size: 18px;
    font-weight: 600;
}
.small-note {
    color: #b0b7c3;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🌐 Network Traffic Load Balancer Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Compare Random, Static, Best Latency, and DQN methods in an interactive dashboard.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    episodes = st.slider("DQN Training Episodes", 10, 100, 20, 10)
    static_path = st.selectbox("Static Path", [0, 1, 2])
    max_steps = st.slider("Max Steps per Run", 10, 100, 50, 10)

    st.markdown("---")
    st.markdown("### 📌 About")
    st.write(
        "This dashboard evaluates four load balancing methods and compares "
        "their total reward and average latency."
    )

    run_button = st.button("🚀 Run Full Comparison", use_container_width=True)

# -----------------------------
# Intro Layout
# -----------------------------
left_intro, right_intro = st.columns([1.4, 1])

with left_intro:
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        This system simulates a <b>network traffic load balancer</b> using multiple approaches:
        <ul>
            <li><b>Random</b> policy</li>
            <li><b>Static</b> policy</li>
            <li><b>Best Latency</b> policy</li>
            <li><b>DQN</b> agent based on Reinforcement Learning</li>
        </ul>
        The goal is to compare how each method performs in terms of:
        <ul>
            <li>Total Reward</li>
            <li>Average Latency</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with right_intro:
    st.markdown('<div class="section-title">Quick Notes</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
        <b>Selected Episodes:</b> {episodes}<br>
        <b>Static Path:</b> {static_path}<br>
        <b>Max Steps:</b> {max_steps}<br><br>
        <span class="small-note">
        Press <b>Run Full Comparison</b> from the sidebar to start the simulation.
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Run Simulation
# -----------------------------
if run_button:
    st.markdown('<div class="section-title">Running Simulation</div>', unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Random
    status_text.info("Running Random policy...")
    env_random = NetworkEnv(max_steps=max_steps)
    r_reward, r_latency = run_random_policy(env_random)
    progress_bar.progress(20)

    # Static
    status_text.info("Running Static policy...")
    env_static = NetworkEnv(max_steps=max_steps)
    s_reward, s_latency = run_static_policy(env_static, fixed_action=static_path)
    progress_bar.progress(40)

    # Best Latency
    status_text.info("Running Best Latency policy...")
    env_best = NetworkEnv(max_steps=max_steps)
    b_reward, b_latency = run_best_latency_policy(env_best)
    progress_bar.progress(60)

    # DQN
    status_text.info("Training DQN agent...")
    env_dqn = NetworkEnv(max_steps=max_steps)
    agent = DQNAgent(state_size=6, action_size=3)

    train_progress = st.progress(0)
    train_status = st.empty()

    for i in range(episodes):
        train_dqn(env_dqn, agent, episodes=1)
        train_progress.progress((i + 1) / episodes)
        train_status.text(f"Training DQN: Episode {i+1}/{episodes}")

    d_reward, d_latency = evaluate_dqn(env_dqn, agent)
    progress_bar.progress(100)
    status_text.success("Simulation completed successfully.")

    # -----------------------------
    # Results Data
    # -----------------------------
    results_df = pd.DataFrame({
        "Method": ["Random", "Static", "Best Latency", "DQN"],
        "Total Reward": [r_reward, s_reward, b_reward, d_reward],
        "Average Latency": [r_latency, s_latency, b_latency, d_latency]
    })

    best_method = results_df.loc[results_df["Total Reward"].idxmax(), "Method"]
    best_reward = results_df["Total Reward"].max()
    best_latency_value = results_df.loc[results_df["Total Reward"].idxmax(), "Average Latency"]

    # -----------------------------
    # Best Method Highlight
    # -----------------------------
    st.markdown('<div class="section-title">Best Performing Method</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="best-box">
        ✅ Best Method: <b>{best_method}</b><br><br>
        Total Reward: <b>{best_reward:.2f}</b><br>
        Average Latency: <b>{best_latency_value:.2f}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # Metrics
    # -----------------------------
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Random Reward", f"{r_reward:.2f}")
        st.metric("Random Latency", f"{r_latency:.2f}")

    with c2:
        st.metric("Static Reward", f"{s_reward:.2f}")
        st.metric("Static Latency", f"{s_latency:.2f}")

    with c3:
        st.metric("Best Latency Reward", f"{b_reward:.2f}")
        st.metric("Best Latency Avg", f"{b_latency:.2f}")

    with c4:
        st.metric("DQN Reward", f"{d_reward:.2f}")
        st.metric("DQN Latency", f"{d_latency:.2f}")

    # -----------------------------
    # Table + Charts
    # -----------------------------
    st.markdown('<div class="section-title">Results Table</div>', unsafe_allow_html=True)
    st.dataframe(results_df, use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("### Total Reward Comparison")
        reward_chart = results_df.set_index("Method")[["Total Reward"]]
        st.bar_chart(reward_chart)

    with chart_col2:
        st.markdown("### Average Latency Comparison")
        latency_chart = results_df.set_index("Method")[["Average Latency"]]
        st.bar_chart(latency_chart)

    # -----------------------------
    # Final Summary
    # -----------------------------
    st.markdown('<div class="section-title">Final Summary</div>', unsafe_allow_html=True)
    st.write(
        f"In this run, **{best_method}** achieved the highest total reward. "
        f"The comparison shows how different load balancing strategies behave under the same simulation setup. "
        f"The DQN method is especially important because it represents the reinforcement learning approach."
    )