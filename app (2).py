import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from PIL import Image

from traffic_env import TrafficEnv


# ─────────────────────────────────────────
# DQN Model (same architecture as train.py)
# ─────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────
env = TrafficEnv()
model = DQN(env.observation_space.shape[0], env.action_space.n)

try:
    model.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=True))
    model.eval()
    MODEL_LOADED = True
except:
    MODEL_LOADED = False


LANE_NAMES = ["North", "South", "East", "West"]
LANE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]


# ─────────────────────────────────────────
# Visualization Function
# ─────────────────────────────────────────
def draw_intersection(queues, green_lane):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_facecolor("#2c3e50")
    fig.patch.set_facecolor("#2c3e50")
    ax.axis("off")

    # Draw roads
    ax.add_patch(plt.Rectangle((3.5, 0), 3, 10, color="#7f8c8d"))
    ax.add_patch(plt.Rectangle((0, 3.5), 10, 3, color="#7f8c8d"))
    ax.add_patch(plt.Rectangle((3.5, 3.5), 3, 3, color="#95a5a6"))

    # Lane positions: North, South, East, West
    positions = [(5, 8.5), (5, 1.5), (8.5, 5), (1.5, 5)]

    for i, (x, y) in enumerate(positions):
        color = "#27ae60" if i == green_lane else "#c0392b"
        # Signal circle
        ax.add_patch(plt.Circle((x, y), 0.6, color=color, zorder=5))
        # Queue bar
        queue_len = int(queues[i])
        ax.text(x, y, str(queue_len), color="white",
                ha="center", va="center", fontsize=11, fontweight="bold", zorder=6)
        ax.text(x, y - 1.0, LANE_NAMES[i], color="white",
                ha="center", va="center", fontsize=8, zorder=6)

    ax.set_title("🚦 Intersection View", color="white", fontsize=13, fontweight="bold")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img


# ─────────────────────────────────────────
# Run Episode Function
# ─────────────────────────────────────────
def run_episode(num_steps):
    if not MODEL_LOADED:
        return None, "Model not loaded. Please upload model.pth to the Space.", ""

    obs, _ = env.reset()
    total_reward = 0
    log_lines = []
    rewards = []

    images = []

    for step in range(int(num_steps)):
        state_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = model(state_t).argmax().item()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(total_reward)

        queues = info["queues"]
        img = draw_intersection(queues, action)
        images.append(img)

        log_lines.append(
            f"Step {step+1:>3} | Green: {LANE_NAMES[action]:<6} | "
            f"Queues: {queues.astype(int).tolist()} | "
            f"Wait: {int(info['total_queue'])}"
        )

        if terminated or truncated:
            break

    # Reward plot
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(rewards, color="#e74c3c", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward over Steps")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    reward_img = Image.open(buf)
    plt.close()

    log_text = "\n".join(log_lines)
    summary = f"Total Reward: {total_reward:.2f} | Steps: {len(log_lines)}"

    return images[-1], log_text, summary, reward_img


# ─────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────
with gr.Blocks(title="Smart Traffic RL", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🚦 Smart Traffic Signal Controller
    ### Deep Reinforcement Learning — DQN Agent
    **Meta Scalar School of Technology Hackathon | Team Nyra**

    An RL agent trained to minimize vehicle waiting time at a 4-way intersection.
    Green circle = lane getting green signal. Numbers = cars waiting.
    """)

    with gr.Row():
        steps_slider = gr.Slider(
            minimum=10, maximum=200, value=50, step=10,
            label="Number of Steps to Simulate"
        )
        run_btn = gr.Button("▶ Run Agent", variant="primary")

    with gr.Row():
        intersection_img = gr.Image(label="Intersection View", width=400)
        reward_plot = gr.Image(label="Reward Progress", width=400)

    summary_text = gr.Textbox(label="Episode Summary", lines=1)
    log_output = gr.Textbox(label="Step-by-step Log", lines=12)

    run_btn.click(
        fn=run_episode,
        inputs=[steps_slider],
        outputs=[intersection_img, log_output, summary_text, reward_plot]
    )

    gr.Markdown("""
    ---
    **How it works:**
    - Environment: 4-lane intersection (Gymnasium format)
    - Agent: Deep Q-Network (DQN) with PyTorch
    - Reward: Negative of normalized total waiting cars
    - Trained over 300 episodes
    """)


if __name__ == "__main__":
    demo.launch()
