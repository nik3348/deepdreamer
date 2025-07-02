import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from model import Model


def prepare_dataset(batch_size, seq_len):
    # Load a small text dataset (using wikitext-2 for example)
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # Set padding token to EOS token
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=seq_len,
            padding='max_length',
            return_tensors=None  # Don't convert to tensors here
        )

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Create DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return dataloader, tokenizer.vocab_size


def train_model():
    # Initialize model
    embedding_dim = 128  # Changed from 256 to 128 to be divisible by 4 heads
    num_attention_heads = 4
    batch_size = 32
    seq_len = 64

    # Prepare dataset
    dataloader, vocab_size = prepare_dataset(batch_size, seq_len)

    model = Model(
        embedding_dim,
        vocab_size,
        num_attention_heads
    )
    model.train()

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    latent_criterion = nn.MSELoss()

    # Training parameters
    num_epochs = 10

    print("Starting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = len(dataloader)

        # Store previous iteration's z_next_pred
        prev_z_pred = None

        for batch_idx, batch in enumerate(dataloader):
            # Get input_ids from the batch and convert to tensor
            input_ids = torch.stack([
                torch.tensor(ids, dtype=torch.long).detach().clone()
                for ids in batch['input_ids']
            ])

            # Forward pass
            x_pred, z, z_next_pred = model(input_ids)

            # Calculate reconstruction loss
            x_pred = x_pred.reshape(-1, vocab_size)
            target_ids = input_ids.reshape(-1)
            recon_loss = criterion(x_pred, target_ids)

            # Calculate next state loss using previous iteration's prediction
            latent_loss = 0
            if prev_z_pred is not None:
                latent_loss = latent_criterion(z, prev_z_pred)

            # Store current z_next_pred for next iteration
            prev_z_pred = z_next_pred.detach()

            # Combined loss
            loss = recon_loss + 0.1 * latent_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
                    f"Latent: {latent_loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(
            f"Epoch {epoch + 1}/{num_epochs} completed. Average loss: {avg_loss:.4f}")

    print("Training completed!")
    return model


if __name__ == "__main__":
    train_model()
