import torch
import datetime
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter

from model import Model

# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
# model = torch.compile(model)


def reward_function(obs):
    return abs(obs[1])


isHuman = True
epochs = 500
batch_size = 32
seq_len = 16
d_model = 64
device = "cuda"
model_location = f"models/model-{d_model}.pt"
log_dir = "logs/fit/" + datetime.datetime.now().strftime(f"%Y-%m-%d.%H:%M.{d_model}")
writer = SummaryWriter(log_dir)
env = gym.make(
    "MountainCar-v0",
    render_mode="human" if isHuman else "rgb_array",
    goal_velocity=0.1,
    max_episode_steps=batch_size * seq_len,
)
action_dim = env.action_space.n
model = Model(action_dim, d_model, batch_size, seq_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

try:
    model.load_state_dict(torch.load(model_location, map_location=device))
    print(f"Model loaded from {model_location}")
except Exception as e:
    print(f"Failed to load model: {e}")

for i in range(epochs):
    obss = []
    z_stack = []
    actions = []
    values = []
    rewards = []
    dones = []
    losses = []

    tokens = torch.stack([torch.tensor(0)]).unsqueeze(0).to(device)
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done or steps < 1000:
        obs = torch.tensor(obs).to(device)
        tokens, logits, value, z, world_loss = model(obs, tokens[:, -(seq_len + 1) :])
        action = model.sample(logits[:, -1, :])

        if False:
            try:
                action = torch.tensor(int(input()))
            except ValueError:
                action = torch.tensor(0)

            action = action.unsqueeze(-1).unsqueeze(-1).to(device)

        obs_next, reward, terminated, truncated, info = env.step(action.item())
        steps += 1
        obs = obs_next
        done = terminated or truncated
        tokens = torch.cat((tokens, action), dim=1)

        obss.append(obs)
        z_stack.append(z)
        actions.append(logits[:, -1, :])
        values.append(value[:, -1, :].item())
        rewards.append(reward + reward_function(obs))
        dones.append(done)
        losses.append(world_loss)

    obss = torch.stack(obss).to(device)
    z_stack = torch.stack(z_stack).to(device)
    actions = torch.stack(actions).to(device)
    values = torch.tensor(values).to(device)
    rewards = torch.tensor(rewards).to(device)
    dones = torch.tensor(dones).to(device)
    losses = torch.tensor(losses).to(device)

    world_loss = losses.mean()
    ae_loss = model.ae_loss(z_stack, obss)
    actor_loss, critic_loss = model.calc_loss(
        actions, rewards, dones, values, ae_loss, world_loss
    )

    writer.add_scalar("Actor Loss", actor_loss, i)
    writer.add_scalar("Critic Loss", critic_loss, i)
    writer.add_scalar("World Model Loss", world_loss, i)
    writer.add_scalar("AE Loss", ae_loss, i)
    writer.add_scalar("Total Loss", actor_loss + critic_loss + world_loss + ae_loss, i)

    if (i + 1) % 10 == 0:
        print(f"Epoch {i + 1} completed")
        torch.save(model.state_dict(), model_location)
        print(f"Model saved to {model_location}")
