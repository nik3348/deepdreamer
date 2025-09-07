import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from datasets import load_dataset
from model import Model


# ----------------------
# Streaming Dataset
# ----------------------
class StreamingTextDataset(IterableDataset):
    """
    A streaming dataset that yields batches of tokenized text, split into fixed-length sequences.
    """

    def __init__(self, dataset, tokenizer, batch_size, seq_len, device, buffer_size=1000):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.buffer_size = buffer_size

    def _process_buffer(self, buffer):
        if not buffer:
            return

        eos_token = self.tokenizer.eos_token or ""
        random.shuffle(buffer)
        joined_text = eos_token.join(buffer)

        encodings = self.tokenizer(
            joined_text,
            add_special_tokens=False,
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].squeeze(0)

        # Trim to a multiple of seq_len
        total_len = (input_ids.shape[0] // self.seq_len) * self.seq_len
        if total_len == 0:
            return

        # Reshape into (num_chunks, seq_len)
        input_ids = input_ids[:total_len].view(-1, self.seq_len)

        # Yield batches
        for i in range(0, input_ids.size(0), self.batch_size):
            batch = input_ids[i:i + self.batch_size].to(self.device)
            yield {'input_ids': batch}

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            buffer.append(sample['text'])
            if len(buffer) >= self.buffer_size:
                yield from self._process_buffer(buffer)
                buffer = []

        # Handle leftover buffer
        if buffer:
            yield from self._process_buffer(buffer)


# ----------------------
# World Model Training
# ----------------------
def recon_training(model, dataloader, global_step, optimizer, num_epochs=2, writer=None, start_epoch=0, checkpoint_path=None):
    """
    Trains the model to reconstruct input sequences and predict latent states.
    Uses DataLoader for efficient batch processing.
    """
    model.train()
    recon_criterion = nn.CrossEntropyLoss()

    if writer is None:
        raise ValueError("A SummaryWriter instance must be provided.")

    print("Starting world model training...")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0

        max_batches = 14000  # or any number you want as a limit
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * batch_size >= max_batches:
                print(
                    f"Reached max_batches ({max_batches}), breaking out of epoch loop.")
                break

            input_ids = batch['input_ids']
            x_pred = model.autoencode(input_ids)

            x_pred = x_pred.reshape(-1, vocab_size)
            x_target = input_ids.reshape(-1)
            recon_loss = recon_criterion(x_pred, x_target)

            # Backpropagation
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()
            total_loss += recon_loss.item()
            torch.cuda.empty_cache()

            # Logging
            global_step += batch['input_ids'].size(0)
            writer.add_scalar('Batch/Recon_Loss',
                              recon_loss.item(), global_step)

            print(
                f"Epoch {epoch + 1}, "
                f"Batch {batch_idx + 1}, "
                f"Recon: {recon_loss.item():.4f}, "
            )

        # Epoch summary
        avg_loss = total_loss / max(batch_idx + 1, 1)
        writer.add_scalar('Epoch/Average_Recon_Loss', avg_loss, epoch)
        print(f"Epoch {epoch + 1} completed. Average Recon loss: {avg_loss:.4f}")

        # Save checkpoint
        if checkpoint_path is not None:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step
            }, checkpoint_path)
            print(
                f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

    print("World model training completed!")
    return model


def latent_training(model, dataloader, global_step, optimizer, num_epochs=2, writer=None, start_epoch=0, checkpoint_path=None):
    """
    Trains the model to reconstruct input sequences and predict latent states.
    Uses DataLoader for efficient batch processing.
    """
    model.train()
    latent_criterion = nn.MSELoss()

    if writer is None:
        raise ValueError("A SummaryWriter instance must be provided.")

    print("Starting world model training...")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0

        max_batches = 1500  # or any number you want as a limit
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * batch_size >= max_batches:
                print(
                    f"Reached max_batches ({max_batches}), breaking out of epoch loop.")
                break

            input_ids = batch['input_ids']
            z, z_next_pred = model.predict_latent(input_ids)

            latent_loss = latent_criterion(
                z_next_pred[:, :-1], z[:, 1:]
            )

            # Backpropagation
            optimizer.zero_grad()
            latent_loss.backward()
            optimizer.step()
            total_loss += latent_loss.item()
            torch.cuda.empty_cache()

            # Logging
            global_step += batch['input_ids'].size(0)
            writer.add_scalar('Batch/Latent_Loss',
                              latent_loss.item(), global_step)

            print(
                f"Epoch {epoch + 1}, "
                f"Batch {batch_idx + 1}, "
                f"Latent: {latent_loss.item():.4f}"
            )

        # Epoch summary
        avg_loss = total_loss / max(batch_idx + 1, 1)
        writer.add_scalar('Epoch/Average_Latent_Loss', avg_loss, epoch)
        print(f"Epoch {epoch + 1} completed. Average Latent loss: {avg_loss:.4f}")

        # Save checkpoint
        if checkpoint_path is not None:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step
            }, checkpoint_path)
            print(
                f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

    print("World model training completed!")
    return model


# ----------------------
# Rollout Training
# ----------------------
def rollout_training(model, dataloader, vocab_size, global_step, optimizer, num_epochs=2, writer=None, T=10, start_epoch=0, checkpoint_path=None, batch_size=16, seq_len=32, tokenizer=None):
    """
    Trains the model to predict future sequences by rolling out in latent space.
    Uses DataLoader for efficient batch processing.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    if writer is None:
        raise ValueError("A SummaryWriter instance must be provided.")

    print("Starting rollout training...")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0

        max_batches = 1000  # or any number you want as a limit
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * batch_size >= max_batches:
                print(
                    f"Reached max_batches ({max_batches}), breaking out of epoch loop.")
                break

            input_ids = batch['input_ids']

            # Encode to latent
            T = 10

            # Rollout
            x_pred = model.rollout(input_ids, T)
            x_pred = x_pred[:, :-T, :].reshape(-1, vocab_size)
            target_ids = input_ids[:, T:].reshape(-1)
            rollout_loss = criterion(x_pred, target_ids)

            # Backpropagation
            optimizer.zero_grad()
            rollout_loss.backward()
            optimizer.step()
            total_loss += rollout_loss.item()
            torch.cuda.empty_cache()

            # Logging
            global_step += batch['input_ids'].size(0)
            writer.add_scalar('Batch/Rollout_Loss',
                              rollout_loss.item(), global_step)

            print(
                f"Epoch {epoch + 1}, "
                f"Batch {batch_idx + 1}, "
                f"Rollout Loss: {rollout_loss.item():.4f}"
            )

        avg_loss = total_loss / max(batch_idx + 1, 1)
        writer.add_scalar('Epoch/Average_Rollout_Loss', avg_loss, epoch)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        # Save checkpoint
        if checkpoint_path is not None:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step
            }, checkpoint_path)
            print(
                f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

    print("Rollout training completed!")
    return model


# ----------------------
# Main Script
# ----------------------
if __name__ == "__main__":
    # Hyperparameters
    embedding_dim = 256
    latent_dim = 256
    num_attention_heads = 16
    num_layers = 18
    batch_size = 8
    seq_len = 20
    global_step = 0
    start_epoch = 0
    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Use streaming with a reasonable sample limit for faster training
    # Load dataset with streaming
    dataset = load_dataset('roneneldan/TinyStories',
                           split='train', streaming=True)
    # Create streaming dataset and dataloader
    streaming_dataset = StreamingTextDataset(
        dataset,
        tokenizer,
        batch_size,
        seq_len,
        device,
        buffer_size=1000  # Add buffer_size for shuffling
    )
    dataloader = DataLoader(
        streaming_dataset,
        batch_size=None,  # batch_size=None for IterableDataset
        num_workers=0,    # Set to 0 for streaming datasets
        pin_memory=False  # Already on device, no need to pin
    )

    model = Model(
        embedding_dim,
        latent_dim,
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
            global_step = checkpoint.get("global_step", 0)
            print("Loaded model and optimizer state.")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state.")

    for _ in range(2):
        # Train world model
        recon_training(
            model,
            dataloader,
            global_step,
            optimizer,
            num_epochs=num_epochs,
            writer=writer,
            start_epoch=start_epoch,
            checkpoint_path=checkpoint_path
        )

        latent_training(
            model,
            dataloader,
            global_step,
            optimizer,
            num_epochs=num_epochs,
            writer=writer,
            start_epoch=start_epoch,
            checkpoint_path=checkpoint_path
        )

        # Train rollout model
        rollout_training(
            model,
            dataloader,
            vocab_size,
            global_step,
            optimizer,
            num_epochs=num_epochs,
            writer=writer,
            start_epoch=start_epoch,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            seq_len=seq_len,
            tokenizer=tokenizer
        )

    # Close writer and save checkpoint
    writer.close()
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": start_epoch + num_epochs,
        "global_step": global_step
    }, checkpoint_path)
