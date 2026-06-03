import numpy.typing as npt
import numpy as np
import torch
import os
import wandb
import typing

from cs336_basics.nn import (
    TransformerLM,
    cross_entropy
)
from cs336_basics.optimizer import (
    AdamW, get_lr_cosin_schedule, gradient_clipping
)

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    rand_ids = torch.randint(len(dataset)-context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[i:i+context_length].astype(np.int64)) for i in rand_ids])
    y = torch.stack([torch.from_numpy(dataset[i+1:i+context_length+1].astype(np.int64)) for i in rand_ids])
    return (x.to(device), y.to(device))

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]

def train(
    vocab_size: int,
    d_model: int,
    d_ff: int,
    rope_theta: float,
    num_heads: int,
    num_layers: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosin_cycle_iters: int,
    batch_size: int,
    context_length: int,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    max_l2_norm: float,
    num_iters: int,
    eval_interval: int,
    eval_iters: int,
    train_data_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    val_data_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    checkpoint_dir: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    restore_checkpoint_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] | None = None,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerLM(
        vocab_size=vocab_size, 
        d_model=d_model, 
        d_ff=d_ff, 
        theta=rope_theta, 
        num_heads=num_heads, 
        num_layers=num_layers, 
        max_seq_len=context_length
    ).to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps
    )

    train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')

    iteration = 0
    if restore_checkpoint_path is not None:
        iteration = load_checkpoint(restore_checkpoint_path, model, optimizer)
        print(f"Resumed training from checkpoint {restore_checkpoint_path}")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    run = wandb.init(
        project="llm_from_scratch",
        config={
            "learning_rate": max_lr,
            "architecture": "TransformerLM",
            "num_heads": num_heads,
            "num_layers": num_layers,
            "context_length": context_length,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "max_l2_norm": max_l2_norm,
        }
    )
    for it in range(iteration, num_iters):
        if it % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = torch.zeros(eval_iters)
                for i in range(eval_iters):
                    x, y = data_loading(val_data, batch_size, context_length, device)
                    predicted_y = model(x)
                    val_losses[i] = cross_entropy(predicted_y, y)
            val_loss = val_losses.mean().item()
            run.log({"val_loss": val_loss}, step=it)
            print(f"Iteration {it}, Val Loss {val_loss}")
            save_checkpoint(model, optimizer, it, os.path.join(checkpoint_dir, f"checkpoint_{it}.pt"))
        model.train()
        lr = get_lr_cosin_schedule(it, max_lr, min_lr, warmup_iters, cosin_cycle_iters)
        for group in optimizer.param_groups:
            group['lr'] = lr

        x, y = data_loading(train_data, batch_size, context_length, device)
        predicted_y = model(x)
        loss = cross_entropy(predicted_y, y)
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm)
        optimizer.step()
        run.log({
            "train_loss": loss.item(), 
            "learning_rate": lr,
            "iteration": it
        }, step=it)
        print(f"Iteration {it}, Loss {loss.item()}")
    run.finish()


    


