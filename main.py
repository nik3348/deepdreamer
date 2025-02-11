import torch
import datetime
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter

from model import Model

# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
# model = torch.compile(model)


def reward_function(obs):
    reward = abs(obs[1])
    if obs[1] > 0:
        reward += 1
    else:
        reward -= 1
    return reward


isHuman = False
epochs = 100
batch_size = 16
seq_len = 16
d_model = 32
exp_rate = 0.01
device = "cuda"
model_location = f"models/model-{d_model}.pt"
log_dir = "logs/fit/" + \
    datetime.datetime.now().strftime(f"%Y-%m-%d.%H:%M.{d_model}")
writer = SummaryWriter(log_dir)
env = gym.make(
    "MountainCar-v0",
    render_mode="human" if isHuman else "rgb_array",
    goal_velocity=0.1,
    max_episode_steps=batch_size * seq_len
)
action_dim = env.action_space.n
model = Model(action_dim, d_model, batch_size, seq_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

try:
    model.load_state_dict(torch.load(model_location, map_location=device))
    print(f"Model loaded from {model_location}")
except:
    pass

for i in range(epochs):
    obss = []
    z_stack = []
    actions = []
    values = []
    rewards = []
    dones = []

    tokens = torch.stack([torch.tensor(0)]).unsqueeze(0).to(device)
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done or steps < 1000:
        obs = torch.tensor(obs).to(device)
        tokens, logits, value, z, world_loss = model(
            obs, tokens[:, -(seq_len + 1):])
        action = model.sample(logits[:, -1, :])

        if torch.rand(1).item() < exp_rate:
            action = torch.randint(0, action_dim, (1, 1), device=device)

        tokens = torch.cat((tokens, action), dim=1)

        obs_next, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        steps += 1

        obss.append(obs)
        z_stack.append(z)
        actions.append(logits[:, -1, :])
        values.append(value[:, -1, :].item())
        rewards.append(reward + reward_function(obs))
        dones.append(done)

        obs = obs_next

        if done:
            break

    obss = torch.stack(obss).to(device)
    z_stack = torch.stack(z_stack).to(device)
    actions = torch.stack(actions).to(device)
    values = torch.tensor(values).to(device)
    rewards = torch.tensor(rewards).to(device)
    dones = torch.tensor(dones).to(device)

    ae_loss = model.ae_loss(z_stack, obss)
    actor_loss, critic_loss = model.calc_loss(
        actions, rewards, dones, values
    )

    writer.add_scalar("Actor Loss", actor_loss, i)
    writer.add_scalar("Critic Loss", critic_loss, i)
    # writer.add_scalar("World Model Loss", world_model_loss, i)
    # writer.add_scalar("Total Loss", actor_loss + critic_loss + world_model_loss, i)

    if (i + 1) % 10 == 0:
        print(f"Epoch {i + 1} completed")
        torch.save(model.state_dict(), model_location)
        print(f"Model saved to {model_location}")
