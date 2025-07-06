import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from datasets import load_dataset
from model import Model


# ----------------------
# World Model Training
# ----------------------
def world_training(model, tokenizer, dataloader, batch_size, optimizer, num_epochs=2, writer=None, device=None, start_epoch=0, checkpoint_path=None):
    """
    Trains the model to reconstruct input sequences and predict latent states.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    latent_criterion = nn.MSELoss()

    if writer is None:
        raise ValueError("A SummaryWriter instance must be provided.")

    print("Starting world model training...")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        num_batches = len(dataloader)
        total_loss = 0
        batch_text = []

        for batch_idx, batch in enumerate(dataloader):
            batch_text.append(batch['text'])
            if (batch_idx + 1) % batch_size != 0:
                continue

            encodings = tokenizer(
                batch_text,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encodings['input_ids'].to(device)
            x = torch.empty(batch_size, 0, dtype=torch.long).to(device)
            z_pred = None

            for i in range(input_ids.size(1)):
                x = torch.cat([x, input_ids[:, i].unsqueeze(1)], dim=1)
                z, z_next_pred, x_pred = model(x)

                x_pred = x_pred.reshape(-1, vocab_size)
                x_target = x.reshape(-1)

                # Compute loss
                recon_loss = criterion(x_pred, x_target)
                if z_pred is not None:
                    latent_loss = latent_criterion(z_pred, z[:, 1:])
                else:
                    # On first iteration, use zero loss for latent component
                    latent_loss = torch.tensor(
                        0.0, device=device, requires_grad=True)

                z_pred = z_next_pred.detach()
                loss = 0.1 * recon_loss + 0.9 * latent_loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                torch.cuda.empty_cache()

            # Logging
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Batch/Loss', loss.item(), global_step)
            writer.add_scalar('Batch/Recon_Loss',
                              recon_loss.item(), global_step)
            writer.add_scalar('Batch/Latent_Loss',
                              latent_loss.item(), global_step)

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{num_batches}, "
                      f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
                      f"Latent: {latent_loss.item():.4f}")

        # Epoch summary
        avg_loss = total_loss / num_batches
        writer.add_scalar('Epoch/Average_Loss', avg_loss, epoch)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        # Save checkpoint
        if checkpoint_path is not None:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1
            }, checkpoint_path)
            print(
                f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

    print("World model training completed!")
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
    batch_size = 64
    seq_len = 64
    start_epoch = 0
    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use streaming with a reasonable sample limit for faster training
    # Load dataset with streaming
    dataset = load_dataset('roneneldan/TinyStories', split='train')

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    model = Model(
        embedding_dim,
        vocab_size,
        num_attention_heads,
        num_layers
    ).to(device)

    checkpoint_path = f"model_{embedding_dim}_{num_attention_heads}_{num_layers}.pt"
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter(
        log_dir=f"runs/training_{embedding_dim}_{num_attention_heads}_{num_layers}")

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
        tokenizer,
        dataset,
        batch_size,
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
        dataset,
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
