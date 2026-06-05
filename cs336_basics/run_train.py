from cs336_basics.training import train

def main():
    train(
        vocab_size=10000,
        d_model=512,
        d_ff=1344,
        rope_theta=10000.0,
        num_heads=16,
        num_layers=4,
        max_lr=6e-4,
        min_lr=6e-5,
        warmup_iters=250,
        cosin_cycle_iters=5000,
        batch_size=32,
        context_length=256,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-8,
        max_l2_norm=1.0,
        num_iters=5000,
        eval_interval=100,
        eval_iters=200,
        train_data_path="train.bin",
        val_data_path="val.bin",
        checkpoint_dir="checkpoint",
        restore_checkpoint_path=None,
    )

if __name__ == "__main__":
    main()
    