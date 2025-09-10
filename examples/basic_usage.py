"""
Basic usage example for DeepSeek V3 model.
"""

import torch
from deepseek import DeepSeekV3, ModelArgs

def main():
    print("DeepSeek V3 From Scratch - Basic Usage Example")
    print("=" * 50)

    # Initialize model with smaller parameters for demonstration
    args = ModelArgs(
        vocab_size=1000,
        dim=512,
        n_layers=2,
        n_heads=8,
        max_seq_len=1024,
        n_routed_experts=8,
        n_activated_experts=2
    )

    # Create model
    model = DeepSeekV3(args)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create sample input
    batch_size, seq_len = 2, 64
    input_tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    print(f"Input shape: {input_tokens.shape}")

    # Inference mode
    model.eval()
    with torch.no_grad():
        logits = model(input_tokens)

    print(f"Output logits shape: {logits.shape}")
    print(f"Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # Training mode
    model.train()
    targets = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    logits, loss = model(input_tokens, targets)

    print(f"Training mode - Logits shape: {logits.shape}")
    print(f"Training loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
