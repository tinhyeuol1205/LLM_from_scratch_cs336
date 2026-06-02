import torch
import numpy as np
from cs336_basics.optimizer import AdamW

def adamw_accounting(batch_size: int, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int):
    total_params = 0
    # Parameters
    phi = 0
    token_emb_param = vocab_size * d_model
    transformer_param = num_layers * (4 * d_model * d_model + 2 * d_model + 3 * d_model * d_ff) + d_model + vocab_size * d_model
    phi += token_emb_param + transformer_param
    total_params += phi
    # Gradient
    grad_param = phi
    total_params += grad_param
    # Optimizer
    adamw_params = 2 * phi
    total_params += adamw_params
    # Activation
    activation_params = batch_size * context_length * d_model * 3 + batch_size * context_length * num_heads * context_length * 2 + batch_size * context_length * d_ff +  2 * batch_size * context_length * d_model + batch_size * context_length * (8/3) * d_model
    

    
    
    
