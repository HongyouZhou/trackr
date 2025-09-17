import torch
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

# Import Flow Matching components
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

        # Load dataset (same as original)
        train_dataset = HumanDataset(
            cfg=config, root=config.pretrain.training.root_dir
        )

        max_ep_len = train_dataset.max_ep_len
        cprint(f"Dataloader built", color="green", attrs=["bold"])

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

        cprint(f"Flow Matching Model built", color="green", attrs=["bold"])

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
        scheduler = None
        optimizer = AdamW(
            model.parameters(),
            lr=config.pretrain.training.lr,
            weight_decay=config.pretrain.training.weight_decay,
        )
        
        # Use Flow Matching specific loss
        loss_fn = torch.nn.L1Loss()  # Can be replaced with FlowMatchingLoss

        # Create Flow Matching trainer
        trainer = FlowMatchingTrainer(
            model=model,
            optimizer=optimizer,
            device=device
        )

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
                
                # Logging
                if batch_idx % config.pretrain.training.log_freq == 0:
                    avg_loss = (sum(train_losses) / max(1, len(train_losses))) if len(train_losses) > 0 else float('nan')
                    cprint(f"Epoch {i}, Batch {batch_idx}, Loss: {avg_loss}", color="cyan")
                    
                    if config.pretrain.wandb_activate:
                        log_dict = {
                            "epoch": i,
                            "batch": batch_idx,
                            "train_loss": avg_loss,
                        }
                        for k in ["proprio_error", "velocity_norm", "conditioning_norm", "xt_norm", "x1_norm"]:
                            val = outputs.get(k)
                            if val is not None and not (isinstance(val, float) and (val != val)):
                                log_dict[k] = val
                        wandb.log(log_dict, commit=True)
            
            # Epoch summary
            epoch_loss = (sum(train_losses) / max(1, len(train_losses))) if len(train_losses) > 0 else float('nan')
            cprint(f"Epoch {i} completed. Average Loss: {epoch_loss}", color="green", attrs=["bold"])
            
            # Save model checkpoint
            if experiment_folder and (i + 1) % config.pretrain.training.model_save_freq == 0:
                checkpoint_path = os.path.join(experiment_folder, f"flow_matching_model_epoch_{i+1}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                cprint(f"Model saved to {checkpoint_path}", color="green")

            # Video capture (if enabled)
            if capture_video and (i + 1) % 10 == 0:  # Capture every 10 epochs
                fps = int(
                    1 / (config.task.sim.dt * config.task.env.controlFrequencyInv)
                )
                print(f"Capturing video from simulation")
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
