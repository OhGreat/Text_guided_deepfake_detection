import argparse

def get_opts() -> argparse.Namespace:
    """
    Parser for all the possible command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    # CLIP generic parameters
    parser.add_argument("--model", type=str, default="ViT-L/14", help="Clip pretrained architecture name.")
    parser.add_argument("--clip_weights", type=str, default="./", help="Path to the original pretrained CLIP weights.")
    parser.add_argument("--topk", type=int, default=None, help="Top k elements to consider during inference of multidescription experiments.")
    
    # CLIP training options
    parser.add_argument("--clip_from_scratch", action='store_true', help="Don't load CLIP pretrained weights when used.")
    parser.add_argument("--contrastive_margin", type=float, default=None, help="Margin for contrastive loss term. If None, no contrastive loss term will be used.")
    parser.add_argument("--only_vision", action='store_true', help="Freeze text encoder and only train vision part of CLIP.")
    parser.add_argument("--reset_vision", action='store_true', help="Resets the vision module pretrained weights.")
    parser.add_argument("--load_only_model", action='store_true', help="Only load the model architecture without pretrained weights.")
    
    # Generic training options
    parser.add_argument("--epochs", type=int, default=1000, help="Maximum number of epochs to train for.")
    parser.add_argument("--early_stop", type=int, default=5, help="Stop the training if the validation loss does not decrease in the number of defined steps.")
    parser.add_argument("--mixed_precision", type=int, default=32, help="32 or 16 bit precision")
    parser.add_argument("--seed", type=int, default=0)

    # Optimizer and scheduler for training options
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate for training.")
    parser.add_argument("--weight_decay", type=float, default=0.2, help="Optimization weight decay.")
    parser.add_argument("--use_scheduler", action='store_true', help="Activates scheduler with warmup and cosine anenaling.")
    parser.add_argument("--sched_step_per_epoch", action='store_true', help="Scheduler step will happen per epoch instead of per batch wneh used.")
    parser.add_argument("--first_cycle_steps", type=int, default=None, help="Number of steps for the first cycle. (Warmup + cosine annealing)")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Number of warmup steps before reaching the target learning rate.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Multiplier for the max learning rate after 1 cycle.")
    parser.add_argument("--min_lr", type=int, default=None, help="Minimum learning rate reachable by scheduler.")
    parser.add_argument("--gradient_clip_val", type=float, default=None, help="Gradient clipping value")

    # Dataset options
    parser.add_argument("--real_train", nargs="+", type=str, default=[], help="List of real faces datasets used for training.")
    parser.add_argument("--fake_train", nargs="+", type=str, default=[], help="List of deepfake datasets used for training.")
    parser.add_argument("--real_eval", nargs="+", type=str, default=[], help="List of real faces datasets used for evaluating.")
    parser.add_argument("--fake_eval", nargs="+", type=str, default=[], help="List of fake faces datasets used for evaluating.")
    parser.add_argument("--real_prompts", type=str, default=None, help="Path to file with real image prompts.")
    parser.add_argument("--fake_prompts", type=str, default=None, help="Path to file with fake image prompts.")
    parser.add_argument("--max_samples_per_class_train", type=int, default=None, help="Limits the number of training samples for each class (real/fake).")
    parser.add_argument("--max_samples_per_class_eval", type=int, default=None, help="Limits the number of evaluation samples for each class (real/fake).")
    parser.add_argument("--max_tot_samples", type=int, default=None, help="Limit the number of samples of the total train dataset with Pytorch Linghting.")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit the number of samples of the total validation dataset with Pytorch Linghting.")
    parser.add_argument("--valid_frac", type=float, default=None, help="Size of the validation split when 'real_eval' and 'fake_eval' are not provided.")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4, help="Number of steps to accumulate gradients over batches.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")

    # Evaluation options
    parser.add_argument("--evaluate", action='store_true', help="Will run evaluation instead of training when defined.")
    parser.add_argument("--eval_file", type=str, default=None, help="Path to csv to save results.")
    parser.add_argument("--data_paths", type=str, default=None, help="Path to the file containing all the data paths in json format for evaluation.")
    parser.add_argument("--interpolate", default="None", help="Interpolate weights between fine tuned and zero shot model for evaluation.")
    
    # Pytorch Ligning extra options (both for training and evaluation)
    parser.add_argument("--no_logger", action='store_true', help="Deactivates logger when used.")
    parser.add_argument("--no_ckpt", action='store_true', help="Deactivates checkpointing when used.")
    parser.add_argument("--log_dir", type=str, default=".", help="Path to save the pytorch lightning_logs folder (Tensorboard).")
    parser.add_argument("--ckpt_path", type=str, default="./TEMP_CKPT", help="Path to save experiment results.")
    parser.add_argument("--ckpt_every_n_steps", type=int, default=None, help="Save a checkpoint every n batch steps.")
    parser.add_argument("--val_check_interval", type=float, default=1., help="Run an evaluation every n steps (as float).")
    parser.add_argument("--limit_train_batches", type=int, default=None, help="Torch lightning argument to limit the number of batches per training epoch.")
    parser.add_argument("--limit_val_batches", type=int, default=None, help="Torch lightning argument to limit the number of batches per evaluation during training.")
    parser.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=0, help="Torch lightning argument to reload dataloaders every n epochs when used.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint from which to continue training.")
    parser.add_argument("--cpu", action='store_true', help="Run on cpu when used.")

    args = parser.parse_args()

    return args