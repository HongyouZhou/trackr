export PYTHONPATH=".:/ssdArray/hongyou/dev/isaacgym/python"
export LD_LIBRARY_PATH="$(which python | sed 's/\/bin\/python//g')/lib:/home/hongyou/miniforge3/envs/unitree-rl/lib"
echo $LD_LIBRARY_PATH
PCKPT=checkpoints/track_predn.pt
CHECKPOINT=$1
CL=8
HIDDEN_DIM=512
N_LAYER=6
N_HEAD=8
cmd="python scripts/distmatch.py num_gpus=1 viser=True \
    task=AllegroXarmThrowing train=AllegroXarmNewPPO_mlp \
    checkpoint=$CHECKPOINT \
    pc_input=True \
    task.env.enableDebugVis=True \
    graphics_device_id=3 \
    train.ppo.learning_rate=1e-4 \
    task.env.input_priv=False \
    test=True headless=False pc_input=True \
    task.env.enableFingertipPosHistory=True \
    task.env.useOldActionSpace=True \
    task.env.useKeypointReward=True \
    task.env.useFingertipReward=False \
    task.env.useFingertipShapeDistReward=False \
    task.env.usePalmReward=False \
    task.env.useHandJointPoseRew=False \
    task.env.useAllegroTips=False \
    task.env.useLiftingReward=False \
    pretrain.checkpoint=$PCKPT \
    pretrain.model.hidden_dim=$HIDDEN_DIM \
    pretrain.model.n_layer=$N_LAYER \
    pretrain.model.n_head=$N_HEAD \
    pretrain.model.context_length=$CL \
    wandb_name=AllegroXarm_MLP \
    rl_device=cuda:0 sim_device=cuda:0 pipeline=gpu \
    train.ppo.minibatch_size=16 num_envs=16"
echo $cmd
eval $cmd