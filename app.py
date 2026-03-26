import streamlit as st
from env import NetworkEnv
from dqn_agent import DQNAgent
from train import train_dqn, run_random_policy, run_static_policy, run_best_latency_policy, evaluate_dqn

st.set_page_config(page_title="Network Load Balancer RL", layout="centered")

st.title("Network Traffic Load Balancer Using Reinforcement Learning")
st.write("This web app allows you to run and compare different load balancing methods.")

method = st.selectbox(
    "Choose a load balancing method",
    ["Random", "Static", "Best Latency", "DQN"]
)

episodes = st.slider("Training Episodes for DQN", 10, 500, 100, 10)
static_path = st.selectbox("Choose Static Path", [0, 1, 2])

if st.button("Run Model"):
    env = NetworkEnv(max_steps=50)

    with st.spinner("Running simulation..."):
        if method == "Random":
            total_reward, avg_latency = run_random_policy(env)

        elif method == "Static":
            total_reward, avg_latency = run_static_policy(env, fixed_action=static_path)

        elif method == "Best Latency":
            total_reward, avg_latency = run_best_latency_policy(env)

        elif method == "DQN":
            agent = DQNAgent(state_size=6, action_size=3)
            train_dqn(env, agent, episodes=episodes)
            total_reward, avg_latency = evaluate_dqn(env, agent)

    st.success(f"{method} completed successfully")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Reward", f"{total_reward:.2f}")
    with col2:
        st.metric("Average Latency", f"{avg_latency:.2f}")