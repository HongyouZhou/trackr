import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint
import wandb
from torch.optim import AdamW
import os
from datetime import datetime
import json
import hydra
from utils.reformat import omegaconf_to_dict
from utils.utils import set_np_formatting, set_seed
from utils.logger import Logger

from algo.tracking.flow_matching_human_model import FlowMatchingHumanModel, FlowMatchingTrainer
from algo.tracking.human_dataset import HumanDataset, collate_fn


@hydra.main(config_name="config", config_path="../cfg/")
def main(config: DictConfig):
    """
    Training script for Flow Matching Human Model
    Based on the original train_tracker.py but adapted for Flow Matching
    """
    
    device = config.pretrain.device
    config.seed = set_seed(config.seed)

    capture_video = config.task.env.enableVideoLog

    if config.pretrain.wandb_activate:
        wandb.init(
            project="manipulation-pretraining-flow-matching",
            name=config.pretrain.wandb_name,
            config=omegaconf_to_dict(config),
        )

    if config.pretrain.test:
        # Test mode - load and evaluate model
        model = FlowMatchingHumanModel(cfg=config)
        model = model.to(device)
        model.eval()

        assert config.pretrain.checkpoint != ""
        set_np_formatting()

        if config.pretrain.wandb_activate:
            wandb_logger = wandb.init(
                project=config.wandb_project,
                name=config.pretrain.wandb_name,
                entity=config.wandb_entity,
                config=omegaconf_to_dict(config),
                sync_tensorboard=True,
            )
        else:
            wandb_logger = None

        output_dif = os.path.join("outputs", config.wandb_name)
        logger = Logger(output_dif, summary_writer=wandb_logger)

        cprint("Start Building the Environment", "green", attrs=["bold"])

        # Import here to avoid circular imports
        from tasks import isaacgym_task_map
        
        env = isaacgym_task_map[config.task_name](
            cfg=omegaconf_to_dict(config.task),
            pretrain_cfg=omegaconf_to_dict(config.pretrain),
            rl_device=config.rl_device,
            sim_device=config.sim_device,
            graphics_device_id=config.graphics_device_id,
            headless=config.headless,
            virtual_screen_capture=config.capture_video,
            force_render=config.force_render,
        )

        model.load_state_dict(
            torch.load(config.pretrain.checkpoint, map_location=device)
        )

        cprint(
            f"Flow Matching Model loaded from {config.pretrain.checkpoint}",
            color="green",
            attrs=["bold"],
        )

        # Note: run_multi_env method needs to be implemented in FlowMatchingHumanModel
        # model.run_multi_env(env, cfg=config)

        return

    else:
        # Training mode
        if config.pretrain.wandb_activate:
            wandb_logger = wandb.init(
                project=config.wandb_project,
                name=config.pretrain.wandb_name,
                entity=config.wandb_entity,
                config=omegaconf_to_dict(config),
            )
        else:
            wandb_logger = None

        # Load datasets with subject-wise split
        train_dataset = HumanDataset(
            cfg=config, 
            root=config.pretrain.training.root_dir,
            split='train',
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=config.seed
        )
        
        val_dataset = HumanDataset(
            cfg=config, 
            root=config.pretrain.training.root_dir,
            split='val',
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=config.seed
        )

        max_ep_len = train_dataset.max_ep_len
        cprint(f"Train dataloader built: {len(train_dataset)} samples", color="green", attrs=["bold"])
        cprint(f"Val dataloader built: {len(val_dataset)} samples", color="green", attrs=["bold"])

        # Create Flow Matching model
        model = FlowMatchingHumanModel(cfg=config, max_ep_len=max_ep_len)
        model = model.to(device)

        # Setup experiment folder
        if config.pretrain.training.model_save_dir is not None:
            save_dir = config.pretrain.training.model_save_dir
            os.makedirs(save_dir, exist_ok=True)
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
            experiment_folder = os.path.join(
                save_dir, f"{config.pretrain.wandb_name}_flow_matching", f"dt_{dt_string}"
            )
            os.makedirs(experiment_folder, exist_ok=True)
            json.dump(
                OmegaConf.to_container(config),
                open(os.path.join(experiment_folder, "config.json"), "w"),
            )
            logger = Logger(experiment_folder, summary_writer=wandb_logger)
        else:
            save_dir = None
            logger = None

        cprint("Flow Matching Model built", color="green", attrs=["bold"])

        # Load checkpoint if specified
        if config.pretrain.training.load_checkpoint:
            assert os.path.exists(
                config.pretrain.checkpoint
            ), f"Checkpoint {config.pretrain.checkpoint} does not exist"
            model.load_state_dict(
                torch.load(config.pretrain.checkpoint, map_location=device)
            )
            model.train()
            cprint(
                f"Flow Matching Model loaded from {config.pretrain.checkpoint}",
                color="green",
                attrs=["bold"],
            )

        # Setup optimizer and loss function
        optimizer = AdamW(
            model.parameters(),
            lr=config.pretrain.training.lr,
            weight_decay=config.pretrain.training.weight_decay,
        )
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.pretrain.training.num_epochs,
            eta_min=config.pretrain.training.lr * 0.01  # Minimum LR is 1% of initial LR
        )
        
        # Create Flow Matching trainer
        trainer = FlowMatchingTrainer(
            model=model,
            optimizer=optimizer,
            device=device
        )
        
        # Create validation dataloader
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.pretrain.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        # Video capture setup (if needed)
        if capture_video:
            assert (
                config.pretrain.wandb_activate
            ), "Video capture requires wandb activation"
            from tasks import isaacgym_task_map
            
            env = isaacgym_task_map[config.task_name](
                cfg=omegaconf_to_dict(config.task),
                pretrain_cfg=omegaconf_to_dict(config.pretrain),
                rl_device=config.pretrain.device,
                sim_device=config.pretrain.device,
                graphics_device_id=config.graphics_device_id,
                headless=config.headless,
                virtual_screen_capture=config.capture_video,
                force_render=config.force_render,
            )

        # Training loop
        for i in range(config.pretrain.training.num_epochs):
            cprint("Flow Matching Training iteration {}".format(i), color="magenta", attrs=["bold"])
            
            # Custom training loop for Flow Matching
            model.train()
            train_losses = []
            
            # Create dataloader
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config.pretrain.training.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0
            )
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
               
                # Train step
                outputs = trainer.train_step(batch)
                # Skip None losses (e.g., NaN batches)
                if outputs.get("loss") is not None and not (isinstance(outputs["loss"], float) and (outputs["loss"] != outputs["loss"])):
                    train_losses.append(outputs["loss"])
                    
                    # Log every step to wandb for smooth loss curve
                    if config.pretrain.wandb_activate:
                        current_loss = outputs["loss"]
                        wandb.log({
                            "train_loss": float(current_loss),
                            "epoch": i,
                            "batch": batch_idx
                        }, commit=False)  # Don't commit yet, wait for epoch end
                
                # Console logging (less frequent)
                if batch_idx % config.pretrain.training.log_freq == 0:
                    avg_loss = (sum(train_losses) / max(1, len(train_losses))) if len(train_losses) > 0 else float('nan')
                    cprint(f"Epoch {i}, Batch {batch_idx}, Loss: {avg_loss}", color="cyan")
                    
                    # Log additional metrics to wandb
                    if config.pretrain.wandb_activate:
                        log_dict = {
                            "epoch": i,
                            "batch": batch_idx,
                        }
                        for metric_name in ["proprio_error", "velocity_norm", "conditioning_norm", "xt_norm", "x1_norm"]:
                            val = outputs.get(metric_name)
                            if val is not None and not (isinstance(val, float) and (val != val)):
                                log_dict[str(metric_name)] = float(val)
                        wandb.log(log_dict, commit=False)  # Don't commit yet
            
            # Epoch summary
            epoch_loss = (sum(train_losses) / max(1, len(train_losses))) if len(train_losses) > 0 else float('nan')
            cprint(f"Epoch {i} completed. Average Train Loss: {epoch_loss}", color="green", attrs=["bold"])
            
            # Validation step
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    # Move batch to device
                    batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
                    
                    # Validation step (same as training but without gradient updates)
                    proprio = batch["hand_kpts"]
                    object_pc = batch["object_pc"]
                    timesteps = batch["timesteps"]
                    labels = batch["labels"]
                    attention_mask = batch["attention_mask"]
                    
                    # Prepare targets
                    proprio_target = torch.clone(proprio[:, 1:])
                    proprio_input = proprio[:, :-1]
                    object_pc_input = object_pc[:, :-1]
                    
                    # Prepare validation data
                    batch_size = proprio_input.shape[0]
                    device_val = proprio_input.device
                    
                    # Sample time for validation (use uniform sampling)
                    t = torch.rand(batch_size, 1, device=device_val)
                    t = t.clamp(1e-4, 1.0 - 1e-4)
                    
                    # Prepare targets
                    kpt_dim = proprio_target.shape[-1]  # Get actual keypoint dimension
                    x_0 = torch.randn(batch_size, kpt_dim, device=device_val)
                    x_1 = proprio_target[:, -1]
                    x_1 = torch.nan_to_num(x_1, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Linear interpolation
                    x_t = (1.0 - t) * x_0 + t * x_1
                    x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Target velocity
                    target_velocity = x_1 - x_0
                    
                    # Get conditioning
                    proprio_input = torch.nan_to_num(proprio_input, nan=0.0, posinf=0.0, neginf=0.0)
                    object_pc_input = torch.nan_to_num(object_pc_input, nan=0.0, posinf=0.0, neginf=0.0)
                    pred_dict, _ = model.forward(
                        proprio_input,
                        object_pc_input,
                        batch["object_ids"],
                        labels,
                        timesteps,
                        attention_mask,
                        batch["object_mask"],
                    )
                    
                    conditioning = pred_dict["conditioning"]
                    predicted_velocity = model.flow_net(x_t, conditioning, t)
                    predicted_velocity = torch.nan_to_num(predicted_velocity, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    val_loss = F.mse_loss(predicted_velocity, target_velocity)
                    if not torch.isnan(val_loss) and not torch.isinf(val_loss):
                        val_losses.append(val_loss.item())
            
            # Calculate average validation loss
            avg_val_loss = (sum(val_losses) / max(1, len(val_losses))) if len(val_losses) > 0 else float('nan')
            cprint(f"Epoch {i} Validation Loss: {avg_val_loss}", color="yellow", attrs=["bold"])
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                if experiment_folder:
                    best_model_path = os.path.join(experiment_folder, "best_flow_matching_model.pt")
                    torch.save(model.state_dict(), best_model_path)
                    cprint(f"New best model saved to {best_model_path}", color="green")
            else:
                patience_counter += 1
                cprint(f"Validation loss did not improve. Patience: {patience_counter}/{patience}", color="red")
            
            # Check for early stopping
            if patience_counter >= patience:
                cprint(f"Early stopping triggered after {i+1} epochs", color="red", attrs=["bold"])
                break
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to wandb and local logger
            log_dict = {
                "epoch": i,
                "train_loss": epoch_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "learning_rate": current_lr
            }
            
            if config.pretrain.wandb_activate:
                wandb.log(log_dict, commit=True)  # Commit all pending logs including step-by-step losses
            
            # Log to local logger if available
            if logger is not None:
                logger.log_scalar("train_loss", epoch_loss, i)
                logger.log_scalar("val_loss", avg_val_loss, i)
                logger.log_scalar("best_val_loss", best_val_loss, i)
                logger.log_scalar("patience_counter", patience_counter, i)
                logger.log_scalar("learning_rate", current_lr, i)
            
            # Save model checkpoint
            if experiment_folder and (i + 1) % config.pretrain.training.model_save_freq == 0:
                checkpoint_path = os.path.join(experiment_folder, f"flow_matching_model_epoch_{i+1}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                cprint(f"Model saved to {checkpoint_path}", color="green")

            # Video capture (if enabled)
            if capture_video and (i + 1) % 10 == 0:  # Capture every 10 epochs
                # fps = int(
                #     1 / (config.task.sim.dt * config.task.env.controlFrequencyInv)
                # )
                print("Capturing video from simulation")
                env.start_video_recording()
                
                # Note: Need to implement run_multi_env for FlowMatchingHumanModel
                # info_dict = model.run_multi_env(env, cfg=config)
                # video_frames = env.stop_video_recording()
                # logger.log_video(
                #     video_frames,
                #     name="Flow Matching Test Performance",
                #     fps=fps,
                #     step=(i + 1) * len(train_dataloader),
                # )
                env.video_frames = []


if __name__ == "__main__":
    main()
