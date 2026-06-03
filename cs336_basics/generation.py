from tokenizer import Tokenizer
from nn import TransformerLM
import torch


def generate(prompt: str, max_new_tokens: int, temperature: float=1.0, top_p: int | None = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerLM(
        vocab_size=10000, 
        d_model=512, 
        d_ff=1344, 
        theta=10000.0, 
        num_heads=16, 
        num_layers=4, 
        max_seq_len=1024
    ).to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    
    tokenizer = Tokenizer.from_files("vocab.json", "merges.json", ["<|endoftext|>"])
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([tokens], device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens[:, -1024:])
            if temperature == 0:
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                logits = logits[:, -1, :] / temperature
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumulative_probs > top_p
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[:, 0] = False
                    sorted_logits[mask] = -float('inf')
                    probs = torch.softmax(sorted_logits, dim=-1)
                    next_token_sorted = torch.multinomial(probs, num_samples=1)
                    next_token = torch.gather(sorted_indices, dim=-1, index=next_token_sorted)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=-1)
            if next_token == tokenizer.vocab["<|endoftext|>"]:
                break
    return tokenizer.decode(tokens[0].tolist())

    

