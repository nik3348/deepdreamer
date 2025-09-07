import torch
from transformers import AutoTokenizer
from model import Model
import os


def load_model(checkpoint_path, embedding_dim, latent_dim, vocab_size, num_attention_heads, num_layers, device):
    model = Model(
        embedding_dim,
        latent_dim,
        vocab_size,
        num_attention_heads,
        num_layers
    ).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    model.eval()
    return model


def generate_text(model, tokenizer, prompt, device, max_length=50):
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids

    for _ in range(max_length):
        # Model expects input shape: (batch, seq_len)
        with torch.no_grad():
            x_pred = model(generated)

            next_token_id = torch.argmax(
                x_pred[:, -1, :], dim=-1)[-1].unsqueeze(0)
            generated = torch.cat(
                (generated, next_token_id.unsqueeze(0)), dim=1)

        # Stop if EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output


if __name__ == "__main__":
    # Hyperparameters (should match those used in training)
    embedding_dim = 256
    latent_dim = 256
    num_attention_heads = 16
    num_layers = 18
    checkpoint_path = f"model_{embedding_dim}_{num_attention_heads}_{num_layers}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    model = load_model(checkpoint_path, embedding_dim, latent_dim,
                       vocab_size, num_attention_heads, num_layers, device)

    # Get prompt from user
    prompt = input("Enter your prompt: ")

    # Generate text
    output = generate_text(model, tokenizer, prompt, device)
    print("\nGenerated text:\n", output)
