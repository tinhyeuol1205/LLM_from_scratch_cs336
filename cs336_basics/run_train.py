from cs336_basics.training import train

def main():
    train(
        vocab_size=50257,
        d_model=768,
        d_ff=3072,
        rope_theta=10000.0,
        num_heads=12,
        num_layers=12,
        max_lr=6e-4,
        min_lr=6e-5,
        warmup_iters=2000,
        cosin_cycle_iters=100000,
        batch_size=32,
        max_tokens=1048576,
        context_length=1024,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-8,
        max_l2_norm=1.0,
        num_iters=100000,
        eval_interval=1000,
        eval_iters=200,
        train_data_path="train.bin",
        val_data_path="val.bin",
        checkpoint_dir="checkpoint",
        restore_checkpoint_path=None,
    )

if __name__ == "__main__":
    main()
    