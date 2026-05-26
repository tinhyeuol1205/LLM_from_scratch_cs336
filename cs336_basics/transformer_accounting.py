import torch

def transformer_acounting(vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int):
    total_params = 0

    # token embedding
    total_params += vocab_size * d_model

    # transformer block
    for _ in range(num_layers):
        # attention mechanism
        total_params += d_model * d_model * 4
        # rms norm
        total_params += 2 * d_model
        # swiglu
        total_params += d_model * d_ff * 3

    # final rms norm
    total_params += d_model
    # output layer
    total_params += d_model * vocab_size
    return total_params

def compute_FLOPs(context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int):
    total_flops = 0

    # transformer block
    for _ in range(num_layers):
        # attention mechanism
        total_flops += context_length * d_model * d_model * 4
        # 
        total_flops += context_length * d_model * context_length
        # swiglu
        total_flops += context_length * d_model * d_ff * 3
    return total_flops

def main():
    print(transformer_acounting(vocab_size=50257, context_length=1024, d_model=1600, num_layers=48, num_heads=25, d_ff=4288))
    print(compute_FLOPs(context_length=1024, d_model=1600, num_layers=48, num_heads=25, d_ff=4288))

if __name__ == "__main__":
    main()

