import os
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
    A streaming dataset that yields batches of tokenized text.
    """

    def __init__(self, dataset, tokenizer, batch_size, seq_len, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

    def __iter__(self):
        batch_texts = []

        for sample in self.dataset:
            batch_texts.append(sample['text'])

            if len(batch_texts) == self.batch_size:
                # Tokenize the batch
                encodings = self.tokenizer(
                    batch_texts,
                    add_special_tokens=False,
                    padding=True,
                    truncation=True,
                    max_length=self.seq_len,
                    return_tensors='pt'
                )

                # Move to device
                encodings['input_ids'] = encodings['input_ids'].to(self.device)
                if 'attention_mask' in encodings:
                    encodings['attention_mask'] = encodings['attention_mask'].to(
                        self.device)

                yield encodings
                batch_texts = []

        # Handle remaining samples
        if batch_texts:
            # Pad with the last sample to reach batch_size if needed
            while len(batch_texts) < self.batch_size:
                batch_texts.append(batch_texts[-1])

            encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=self.seq_len,
                return_tensors='pt'
            )

            encodings['input_ids'] = encodings['input_ids'].to(self.device)
            if 'attention_mask' in encodings:
                encodings['attention_mask'] = encodings['attention_mask'].to(
                    self.device)

            yield encodings


# ----------------------
# World Model Training
# ----------------------
def compute_sequence_losses(model, input_ids, batch_size, recon_criterion, latent_criterion, device, vocab_size, seq_len):
    """
    Compute reconstruction and latent losses for a sequence of tokens,
    using a sliding window of fixed length.

    Args:
        model: The model to use for forward pass
        input_ids: Input token IDs tensor
        batch_size: Size of the batch
        recon_criterion: Loss recon_criterion for reconstruction
        latent_criterion: Loss recon_criterion for latent states
        device: Device to run computations on
        vocab_size: Size of the vocabulary
        seq_len: Maximum sequence length window to keep during input

    Returns:
        tuple: (recon_loss, latent_loss)
    """
    x = torch.empty(batch_size, 0, dtype=torch.long).to(device)
    z_pred = None
    recon_losses = []
    latent_losses = []

    for i in range(input_ids.size(1)):
        x = torch.cat([x, input_ids[:, i].unsqueeze(1)], dim=1)
        if x.size(1) > seq_len:
            x = x[:, -seq_len:]
            z_pred = z_pred[:, -seq_len+1:, :]

        z, z_next_pred, x_pred = model(x)

        x_pred = x_pred.reshape(-1, vocab_size)
        x_target = x.reshape(-1)
        recon_loss = recon_criterion(x_pred, x_target)

        if z_pred is not None:
            latent_loss = latent_criterion(z_pred, z[:, 1:].detach())
        else:
            latent_loss = torch.tensor(0.0, device=device, requires_grad=True)

        z_pred = z_next_pred
        recon_losses.append(recon_loss)
        latent_losses.append(latent_loss)

    recon_loss = torch.stack(recon_losses).mean()
    latent_loss = torch.stack(latent_losses).mean()

    return recon_loss, latent_loss


def world_training(model, dataloader, batch_size, seq_len, global_step, optimizer, num_epochs=2, writer=None, device=None, start_epoch=0, checkpoint_path=None):
    """
    Trains the model to reconstruct input sequences and predict latent states.
    Uses DataLoader for efficient batch processing.
    """
    model.train()
    recon_criterion = nn.CrossEntropyLoss()
    latent_criterion = nn.MSELoss()

    if writer is None:
        raise ValueError("A SummaryWriter instance must be provided.")

    print("Starting world model training...")
    for epoch in range(start_epoch, start_epoch + num_epochs):
        total_loss = 0
        batch_count = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids']

            recon_loss, latent_loss = compute_sequence_losses(
                model,
                input_ids,
                batch_size,
                recon_criterion,
                latent_criterion,
                device,
                vocab_size,
                seq_len
            )
            loss = 0.2 * recon_loss + 0.8 * latent_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            torch.cuda.empty_cache()

            # Logging
            global_step += 1
            batch_count += 1
            writer.add_scalar('Batch/Loss', loss.item(), global_step)
            writer.add_scalar('Batch/Recon_Loss',
                              recon_loss.item(), global_step)
            writer.add_scalar('Batch/Latent_Loss',
                              latent_loss.item(), global_step)

            print(
                f"Epoch {epoch + 1}, "
                f"Batch {batch_count}, "
                f"Loss: {loss.item():.4f}, "
                f"Recon: {recon_loss.item():.4f}, "
                f"Latent: {latent_loss.item():.4f}"
            )

        # Epoch summary
        avg_loss = total_loss / max(batch_count, 1)
        writer.add_scalar('Epoch/Average_Loss', avg_loss, epoch)
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

    print("World model training completed!")
    return model


# ----------------------
# Rollout Training
# ----------------------
def rollout_training(model, dataloader, vocab_size, global_step, optimizer, num_epochs=2, writer=None, device=None, T=4, start_epoch=0, checkpoint_path=None, batch_size=16, seq_len=32, tokenizer=None):
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
        batch_count = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids']

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
            torch.cuda.empty_cache()

            # Logging
            global_step += 1
            batch_count += 1
            writer.add_scalar('Batch/Rollout_Loss',
                              rollout_loss.item(), global_step)

            print(
                f"Epoch {epoch + 1}, "
                f"Batch {batch_count}, "
                f"Rollout Loss: {rollout_loss.item():.4f}"
            )

        avg_loss = total_loss / max(batch_count, 1)
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
    embedding_dim = 128
    num_attention_heads = 8
    num_layers = 8
    batch_size = 16
    seq_len = 32
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
        device
    )
    dataloader = DataLoader(
        streaming_dataset,
        batch_size=None,  # batch_size=None for IterableDataset
        num_workers=0,    # Set to 0 for streaming datasets
        pin_memory=False  # Already on device, no need to pin
    )

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
            global_step = checkpoint.get("global_step", 0)
            print("Loaded model and optimizer state.")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state.")

    # Train world model
    world_training(
        model,
        dataloader,
        batch_size,
        seq_len,
        global_step,
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
        global_step,
        optimizer,
        num_epochs=num_epochs,
        writer=writer,
        device=device,
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
