#!/bin/bash
TASK=reach # place grasp reach
ENCODER=dinov2 # dinov2
NODES=1 # number of nodes to use for distributed testing
DATA_DIR=/home/yulun/projects/dexwm/data

# Parse command line arguments
ENV_FLAGS=""
USE_SUBMITIT=0
QOS=explore
TIMEOUT=300

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --encoder)
      ENCODER="$2"
      shift 2
      ;;
    --kp_loss_weight)
      KP_LOSS_WEIGHT="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --nodes)
      NODES="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --debug)
      DEBUG_FLAG="--debug"
      shift
      ;;
    --render)
      RENDER_FLAG="--render"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      shift
      ;;
  esac
done

# Set encoder-specific variables
if [ "$ENCODER" == "dinov2" ]; then
  CKPT_PATH=/home/yulun/projects/dexwm/data/checkpoints/egodex_0.pth_24000.tar
  CONFIG=configs/egodex.yaml
  DEC_CKPT=""
  KP_LOSS_WEIGHT=${KP_LOSS_WEIGHT:-1e-3}
fi

# Update paths after encoder is set
JOB_DIR=/home/yulun/projects/dexwm/output/robocasa_eval/${TASK}_${KP_LOSS_WEIGHT}
# JOB_DIR=debug3

# Set task-specific flags
TASK_FLAGS=""
DEBUG_FLAG=${DEBUG_FLAG:-""}
RENDER_FLAG=${RENDER_FLAG:-""}
if [ "$TASK" == "grasp" ]; then
  TASK_FLAGS="--use_loss_ee"
fi

echo $JOB_DIR
echo $TASK
echo $ENCODER
echo $TASK_FLAGS
echo $KP_LOSS_WEIGHT
echo "Nodes: $NODES"

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"
conda activate robot_sim_dexwm

DEC_CKPT_FLAG=""
if [ -n "$DEC_CKPT" ]; then
  DEC_CKPT_FLAG="--decoder_checkpoint $DEC_CKPT"
fi

if [ "$USE_SUBMITIT" -eq 0 ]; then
  torchrun --nproc-per-node 1 --nnodes 1 \
    -m sim_test \
    --job_dir $JOB_DIR \
    --config $CONFIG \
    --checkpoint $CKPT_PATH \
    --method dexwm \
    --task_name $TASK \
    --kp_loss_weight $KP_LOSS_WEIGHT \
    --data_dir $DATA_DIR \
    --num_samples 16 \
    --batch_size 2 \
    --topk 4 \
    --pred_steps 1 \
    $DEC_CKPT_FLAG \
    $DEBUG_FLAG \
    $RENDER_FLAG \
    $TASK_FLAGS
else
  eval "$ENV_FLAGS python -m submit_test_sim_task --nodes $NODES --cpus_per_task 16 --timeout $TIMEOUT --partition learn --qos $QOS \
        --job_dir $JOB_DIR \
        --config $CONFIG \
        --checkpoint $CKPT_PATH \
        --method dexwm \
        --task_name $TASK \
        --kp_loss_weight $KP_LOSS_WEIGHT \
        --data_dir $DATA_DIR \
        --decoder_checkpoint $DEC_CKPT \
        $TASK_FLAGS"
fi
