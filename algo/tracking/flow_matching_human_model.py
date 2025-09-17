import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingHumanModel(nn.Module):
    """
    Flow Matching based Human Model for hand motion prediction
    Replaces Transformer with Flow Matching for better continuous motion generation
    """
    
    def __init__(self, cfg, max_ep_len=4096):
        super().__init__()
        
        self.kpt_dim = 5 * 3  # 5 fingertips * 3D coordinates
        self.pc_num = 100
        self.total_obj_num = 22
        self.max_ep_len = max_ep_len
        self.device = cfg.pretrain.device
        self.max_obj_num = 5
        self.n_ctx = (
            self.max_obj_num + 1
        ) * cfg.pretrain.model.context_length + 1  # +1 for the object id
        self.hidden_size = cfg.pretrain.model.hidden_dim
        self.cfg = cfg
        
        # Flow Matching parameters
        self.num_inference_steps = getattr(cfg.pretrain.model, 'num_inference_steps', 50)
        self.time_sampling_strategy = getattr(cfg.pretrain.model, 'time_sampling_strategy', 'uniform')
        self.max_velocity_magnitude = getattr(cfg.pretrain.model, 'max_velocity_magnitude', None)
        
        # Multi-modal encoders (与原始HumanModel保持一致)
        self.embed_proprio = torch.nn.Linear(self.kpt_dim, self.hidden_size)
        
        self.embed_pc = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.MaxPool2d((self.pc_num, 1)),
        )  # PointNet
        
        self.embed_timestep = nn.Embedding(
            self.n_ctx, self.hidden_size
        )  # 1 extra for padding
        
        self.embed_label = nn.Embedding(self.total_obj_num, self.hidden_size)
        
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        
        # Flow Matching specific components
        self.flow_net = FlowMatchingNetwork(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size * 2,
            output_dim=self.kpt_dim,
            num_layers=6
        )
        # propagate optional cap to the flow net
        self.flow_net.max_velocity_magnitude = self.max_velocity_magnitude
        
        # Multi-modal fusion network
        self.fusion_net = MultiModalFusion(
            proprio_dim=self.hidden_size,
            pc_dim=self.hidden_size,
            time_dim=self.hidden_size,
            label_dim=self.hidden_size,
            output_dim=self.hidden_size
        )
        
        # Context encoder for sequence modeling
        self.context_encoder = ContextEncoder(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            num_layers=4
        )
        
        # Output projection (与原始HumanModel保持一致)
        self.predict_proprio = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.kpt_dim)])
        )
        
    def forward(self, proprio, object_pc, object_ids, labels, timesteps, attention_mask, object_mask=None):
        """
        Forward pass with Flow Matching
        """
        batch_size = proprio.shape[0]
        
        # Encode multi-modal inputs (与原始HumanModel保持一致)
        proprio_embeddings = self.embed_proprio(proprio)
        pc_embeddings = self.embed_pc(
            object_pc.reshape(batch_size, -1, *object_pc.shape[-2:])
        ).squeeze(-2)
        pc_embeddings = pc_embeddings.reshape(
            batch_size, object_pc.shape[1], object_pc.shape[2], -1
        )
        time_embeddings = self.embed_timestep(timesteps)
        label_embeddings = self.embed_label(labels)
        
        # Add time embeddings
        proprio_embeddings = proprio_embeddings + time_embeddings
        pc_embeddings = pc_embeddings + time_embeddings.unsqueeze(-2)
        pc_embeddings = pc_embeddings + self.embed_label(object_ids).unsqueeze(1)
        
        # Multi-modal fusion
        fused_features = self.fusion_net(
            proprio_embeddings, pc_embeddings, time_embeddings, label_embeddings
        )
        # Normalize fused features for stability
        fused_features = self.embed_ln(fused_features)
        
        # Context encoding for sequence modeling
        context_features = self.context_encoder(fused_features, attention_mask)
        
        # Flow Matching prediction
        # 使用整个序列的信息作为条件，而不是只使用最后一个时间步
        conditioning = context_features.mean(dim=1)  # (batch_size, hidden_size)
        # 或者使用最后一个时间步: conditioning = context_features[:, -1] ?
        
        # 在训练时，只返回条件信息，不进行预测
        # 在推理时，使用Flow Matching生成预测
        if self.training:
            pred_dict = {"conditioning": conditioning}
        else:
            # 推理时使用FM采样（状态依赖速度场积分）
            next_kpt_preds = self.flow_net.sample(conditioning, num_steps=self.num_inference_steps)
            pred_dict = {"next_proprio": next_kpt_preds}
        
        return pred_dict, pc_embeddings
    
    def sample_trajectory(self, conditioning, num_steps=None):
        """
        Sample a trajectory using Flow Matching
        """
        if num_steps is None:
            num_steps = self.num_inference_steps
        return self.flow_net.sample(conditioning, num_steps)


class FlowMatchingNetwork(nn.Module):
    """
    Flow Matching network for continuous motion generation
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Optional safety cap on predicted velocity magnitude (can be overridden via model holder)
        self.max_velocity_magnitude = None
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # Trunk network (state + conditioning + time embedding -> hidden)
        trunk_layers = []
        # 输入拼接: [x_t (output_dim), conditioning (input_dim), t_embed (hidden_dim//2)]
        self.input_concat_dim = input_dim + output_dim + hidden_dim // 2
        self.input_norm = nn.LayerNorm(self.input_concat_dim)
        trunk_layers.append(nn.Linear(self.input_concat_dim, hidden_dim))
        trunk_layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(nn.SiLU())
            
        self.trunk = nn.Sequential(*trunk_layers)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        
        # Direction-magnitude heads for stable velocity prediction
        self.direction_head = nn.Linear(hidden_dim, output_dim)
        self.magnitude_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_t, conditioning, t=None):
        """
        Forward pass for training
        """
        if t is None:
            # For inference, use t=1.0 (end of flow)
            t = torch.ones(conditioning.shape[0], 1, device=conditioning.device)
        else:
            # Clamp time to [0,1] to avoid out-of-range embeddings/numerics
            t = t.clamp(0.0, 1.0)
        
        # Time embedding
        t_embed = self.time_embed(t)
        
        # Concatenate current state, conditioning and time
        net_in = torch.cat([x_t, conditioning, t_embed], dim=-1)
        net_in = self.input_norm(net_in)
        
        # Trunk forward
        h = self.trunk(net_in)
        h = self.hidden_norm(h)
        
        # Direction-magnitude parameterization
        dir_raw = self.direction_head(h)
        mag_raw = self.magnitude_head(h)
        
        dir_norm = torch.norm(dir_raw, dim=-1, keepdim=True).clamp_min(1e-6)
        direction = dir_raw / dir_norm
        magnitude = F.softplus(mag_raw)
        # Optional clamp for numerical stability
        if self.max_velocity_magnitude is not None:
            magnitude = magnitude.clamp(max=self.max_velocity_magnitude)
        
        velocity = magnitude * direction
        return velocity
    
    def sample(self, conditioning, num_steps=50):
        """
        Sample from the flow
        """
        batch_size = conditioning.shape[0]
        device = conditioning.device
        
        # Start from noise
        x = torch.randn(batch_size, self.output_dim, device=device)
        
        # Euler integration
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=device)
            velocity = self.forward(x, conditioning, t)
            x = x + dt * velocity
            
        return x


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion network
    """
    
    def __init__(self, proprio_dim, pc_dim, time_dim, label_dim, output_dim):
        super().__init__()
        
        self.proprio_proj = nn.Linear(proprio_dim, output_dim)
        self.pc_proj = nn.Linear(pc_dim, output_dim)
        self.time_proj = nn.Linear(time_dim, output_dim)
        self.label_proj = nn.Linear(label_dim, output_dim)
        
        # Cross-attention for fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
        
    def forward(self, proprio_emb, pc_emb, time_emb, label_emb):
        """
        Fuse multi-modal features
        """
        # Project to common dimension
        proprio_proj = self.proprio_proj(proprio_emb)  # (batch_size, seq_length, hidden_size)
        pc_proj = self.pc_proj(pc_emb.mean(dim=2))     # (batch_size, seq_length, hidden_size)
        time_proj = self.time_proj(time_emb)           # (batch_size, seq_length, hidden_size)
        
        # 将标签嵌入扩展到序列长度
        label_proj = self.label_proj(label_emb).unsqueeze(1).expand(-1, proprio_emb.shape[1], -1)
        # (batch_size, seq_length, hidden_size)
        
        # 融合多模态特征
        fused = proprio_proj + pc_proj + time_proj + label_proj
        # (batch_size, seq_length, hidden_size)
        
        # 序列注意力
        attended, _ = self.cross_attn(fused, fused, fused)
        # (batch_size, seq_length, hidden_size)
        
        # 最终融合
        output = self.fusion_mlp(attended)
        # (batch_size, seq_length, hidden_size)
        
        return output


class ContextEncoder(nn.Module):
    """
    Context encoder for sequence modeling
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=4):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x, attention_mask):
        """
        Encode sequence context
        """
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~attention_mask.bool())
        return x


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching loss function
    """
    
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, predicted_velocity, target_velocity, t):
        """
        Compute Flow Matching loss
        """
        # Simple L2 loss for now
        # In practice, you might want to use more sophisticated loss functions
        loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss


# Training wrapper
class FlowMatchingTrainer:
    """
    Training wrapper for Flow Matching Human Model
    """
    
    def __init__(self, model, optimizer, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = FlowMatchingLoss()
        self.grad_clip_norm = 1.0
        
    def _sample_time(self, batch_size, device):
        """
        Sample time steps based on configured strategy
        """
        strategy = self.model.time_sampling_strategy
        
        if strategy == 'uniform':
            # Uniform sampling: t ~ U(0, 1)
            return torch.rand(batch_size, 1, device=device)
        elif strategy == 'beta':
            # Beta distribution: t ~ Beta(2, 2) - more samples near 0.5
            import torch.distributions as dist
            beta_dist = dist.Beta(2.0, 2.0)
            return beta_dist.sample((batch_size, 1)).to(device)
        elif strategy == 'cosine':
            # Cosine schedule: more samples near t=0 and t=1
            u = torch.rand(batch_size, 1, device=device)
            return 0.5 * (1 - torch.cos(u * torch.pi))
        elif strategy == 'log_uniform':
            # Log-uniform sampling: more samples near t=0
            u = torch.rand(batch_size, 1, device=device)
            return torch.exp(u * torch.log(torch.tensor(1e-4, device=device))) * (1 - 1e-4) + 1e-4
        else:
            # Default to uniform
            return torch.rand(batch_size, 1, device=device)
        
    def train_step(self, batch):
        """
        Training step for Flow Matching
        """
        proprio = batch["hand_kpts"]
        object_pc = batch["object_pc"]
        timesteps = batch["timesteps"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        
        # Prepare targets
        proprio_target = torch.clone(proprio[:, 1:])
        proprio_input = proprio[:, :-1]
        object_pc_input = object_pc[:, :-1]
        
        # 准备Flow Matching训练数据
        batch_size = proprio_input.shape[0]
        device = proprio_input.device
        
        # 采样Flow Matching的时间步 t
        t = self._sample_time(batch_size, device)
        t = t.clamp(1e-4, 1.0 - 1e-4)
        
        # 准备目标：从噪声到真实手部关键点的流
        # x_0: 噪声 (batch_size, 15)
        x_0 = torch.randn(batch_size, 15, device=device)
        # x_1: 真实的下一个手部关键点 (batch_size, 15)
        x_1 = proprio_target[:, -1]
        # Remove explicit clamping, keep NaN/Inf sanitization only
        x_1 = torch.nan_to_num(x_1, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 线性插值：x_t = (1-t) * x_0 + t * x_1（用于对任意中间状态的监督）
        x_t = (1.0 - t) * x_0 + t * x_1
        x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算目标速度场：v_t = x_1 - x_0
        target_velocity = x_1 - x_0
        
        # 获取条件信息
        # Sanitize model inputs
        proprio_input = torch.nan_to_num(proprio_input, nan=0.0, posinf=0.0, neginf=0.0)
        object_pc_input = torch.nan_to_num(object_pc_input, nan=0.0, posinf=0.0, neginf=0.0)
        pred_dict, _ = self.model.forward(
            proprio_input,
            object_pc_input,
            batch["object_ids"],
            labels,
            timesteps,
            attention_mask,
            batch["object_mask"],
        )
        
        # 使用Flow Matching网络预测速度场
        conditioning = pred_dict["conditioning"]  # 需要从模型中获取条件
        predicted_velocity = self.model.flow_net(x_t, conditioning, t)
        predicted_velocity = torch.nan_to_num(predicted_velocity, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算Flow Matching损失
        loss = F.mse_loss(predicted_velocity, target_velocity)
        if torch.isnan(loss) or torch.isinf(loss):
            # Skip this batch to avoid poisoning the optimizer state
            self.optimizer.zero_grad(set_to_none=True)
            return {
                "loss": None,
                "proprio_error": None,
                "skipped": True,
            }
        
        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        
        # Diagnostics
        vel_norm = predicted_velocity.norm(dim=-1).mean().detach().cpu().item()
        cond_norm = conditioning.norm(dim=-1).mean().detach().cpu().item()
        xt_norm = x_t.norm(dim=-1).mean().detach().cpu().item()
        x1_norm = x_1.norm(dim=-1).mean().detach().cpu().item()
        return {
            "loss": loss.detach().cpu().item(),
            "proprio_error": loss.detach().cpu().item(),
            "velocity_norm": vel_norm,
            "conditioning_norm": cond_norm,
            "xt_norm": xt_norm,
            "x1_norm": x1_norm,
        }
