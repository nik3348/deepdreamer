import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from datasets import load_dataset
from model import Model


# ----------------------
# Sliding Window Chunking
# ----------------------
class SlidingWindowDataset(Dataset):
    def __init__(self, token_ids, seq_len=128, stride=64):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.stride = stride
        self.samples = self._create_chunks()

    def _create_chunks(self):
        chunks = []
        for i in range(0, len(self.token_ids) - self.seq_len + 1, self.stride):
            chunks.append(self.token_ids[i:i + self.seq_len])
        return chunks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.samples[idx], dtype=torch.long)}


# ----------------------
# Dataset Preparation
# ----------------------
def prepare_dataset(batch_size, seq_len, stride=32):
    """
    Loads and tokenizes the wikitext-2 dataset for language modeling.
    Splits into overlapping chunks using a sliding window.
    Returns a DataLoader and the vocabulary size.
    """
    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize each line separately and flatten to a single list of token IDs
    all_texts = [line for line in dataset['text'] if line.strip()]
    token_ids = []
    for text in all_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.extend(ids)

    # Create sliding window dataset
    sw_dataset = SlidingWindowDataset(
        token_ids, seq_len=seq_len, stride=stride
    )

    # Create dataloader
    dataloader = DataLoader(
        sw_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return dataloader, tokenizer.vocab_size


# ----------------------
# World Model Training
# ----------------------
def world_training(model, dataloader, vocab_size, optimizer, num_epochs=2, writer=None, device=None, start_epoch=0, checkpoint_path=None):
    """
    Trains the model to reconstruct input sequences and predict latent states.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    latent_criterion = nn.MSELoss()
    if writer is None:
        raise ValueError("A SummaryWriter instance must be provided.")

    print("Starting training...")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0
        num_batches = len(dataloader)
        prev_z_pred = None
        for batch_idx, batch in enumerate(dataloader):
            # Prepare input tensor
            input_ids = torch.stack([
                ids.detach().clone().long()
                for ids in batch['input_ids']
            ]).to(device)

            # Forward pass
            z, z_next_pred, x_pred = model(input_ids)
            x_pred = x_pred.reshape(-1, vocab_size)
            target_ids = input_ids.reshape(-1)

            # Compute losses
            recon_loss = criterion(x_pred, target_ids)
            latent_loss = 0.0

            if prev_z_pred is not None:
                latent_loss = latent_criterion(z, prev_z_pred)
                latent_loss = latent_loss.item()

            prev_z_pred = z_next_pred.detach()
            loss = recon_loss + 0.5 * latent_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Logging
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Batch/Loss', loss.item(), global_step)
            writer.add_scalar('Batch/Recon_Loss', recon_loss, global_step)
            writer.add_scalar('Batch/Latent_Loss', latent_loss, global_step)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}, "
                    f"Batch {batch_idx + 1}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Recon: {recon_loss:.4f}, "
                    f"Latent: {latent_loss:.4f}"
                )

        avg_loss = total_loss / num_batches
        writer.add_scalar('Epoch/Average_Loss', avg_loss, epoch)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        # Save checkpoint every epoch
        if checkpoint_path is not None:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1
            }, checkpoint_path)
            print(
                f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

    print("Training completed!")
    return model


# ----------------------
# Rollout Training
# ----------------------
def rollout_training(model, dataloader, vocab_size, optimizer, num_epochs=2, writer=None, device=None, T=4, start_epoch=0, checkpoint_path=None):
    """
    Trains the model to predict future sequences by rolling out in latent space.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    if writer is None:
        raise ValueError("A SummaryWriter instance must be provided.")

    print("Starting training...")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0
        num_batches = len(dataloader)
        for batch_idx, batch in enumerate(dataloader):
            # Prepare input tensor
            input_ids = torch.stack([
                ids.detach().clone().long()
                for ids in batch['input_ids']
            ]).to(device)

            # Encode to latent
            z = model.encode(input_ids)
            rollout_loss = 0
            for _ in range(T):
                # Rollout latent future and predict next tokens
                z, x_pred = model.rollout_latent_future(z)
                x_pred = x_pred[:, :-1, :].reshape(-1, vocab_size)
                target_ids = input_ids[:, 1:].reshape(-1)
                rollout_loss += criterion(x_pred, target_ids)

            # Backpropagation
            optimizer.zero_grad()
            rollout_loss.backward()
            optimizer.step()
            total_loss += rollout_loss.item()

            # Logging
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Batch/Rollout_Loss',
                              rollout_loss.item(), global_step)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}, "
                    f"Batch {batch_idx + 1}/{num_batches}, "
                    f"Rollout Loss: {rollout_loss.item():.4f}, ")

        avg_loss = total_loss / num_batches
        writer.add_scalar('Epoch/Average_Rollout_Loss', avg_loss, epoch)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        # Save checkpoint every epochs
        if checkpoint_path is not None:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1
            }, checkpoint_path)
            print(
                f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

    print("Training completed!")
    return model


# ----------------------
# Main Script
# ----------------------
if __name__ == "__main__":
    # Hyperparameters
    embedding_dim = 128
    num_attention_heads = 8
    num_layers = 8
    batch_size = 32
    seq_len = 64
    start_epoch = 0
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader, vocab_size = prepare_dataset(batch_size, seq_len)
    model = Model(
        embedding_dim,
        vocab_size,
        num_attention_heads,
        num_layers
    ).to(device)

    checkpoint_path = f"model_{embedding_dim}.pt"
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter(log_dir="runs/training")

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            print("Loaded model and optimizer state.")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state.")

    # Train world model
    world_training(
        model,
        dataloader,
        vocab_size,
        optimizer,
        num_epochs=num_epochs,
        writer=writer,
        device=device,
        start_epoch=start_epoch,
        checkpoint_path=checkpoint_path
    )

    # Train rollout model
    rollout_training(
        model,
        dataloader,
        vocab_size,
        optimizer,
        num_epochs=num_epochs,
        writer=writer,
        device=device,
        start_epoch=start_epoch,
        checkpoint_path=checkpoint_path
    )

    # Close writer and save checkpoint
    writer.close()
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": start_epoch + num_epochs
    }, checkpoint_path)
