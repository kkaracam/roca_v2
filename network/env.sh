# System config
export OMP_NUM_THREADS=1
export NUM_WORKERS=4
export SEED=2021
export CUDA_VISIBLE_DEVICES=2

# NOTE: Change the data config based on your setup!
# JSON files
export DATA_DIR=/mnt/noraid/karacam/Roca/Data/Dataset/CustomCenter
# Resized images with intrinsics and poses
export IMAGE_ROOT=/mnt/noraid/karacam/Roca/Data/Images #/mnt/noraid/karacam/AlignmentNew/Resized400k/tasks/scannet_frames_25k #$HOME/Thesis/ROCA/Data/Images
# Depths and instances rendered over images
export RENDERING_ROOT=/mnt/noraid/karacam/Roca/Data/Rendering #/mnt/noraid/karacam/AlignmentNew/Resized400k/tasks/scannet_frames_25k #$HOME/Thesis/ROCA/Data/Rendering
# Scan2CAD Full Annotations
export FULL_ANNOT=/mnt/noraid/karacam/Roca/Data/Dataset/CustomCenter/full_annotations_centered.json #$HOME/Thesis/ROCA/Data/full_annotations.json

# Model configurations
export RETRIEVAL_MODE="joint" #joint+comp, resnet_resnet+image+comp, nearest
export RETRIEVAL_LOSS="regression" #bce, triplet
export E2E=1
export NOC_WEIGHTS=1

export JOINT_MODEL_PATH="/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/log/chair522_1024p_v1_icp/model.pth"

# Train and test behavior
export EVAL_ONLY=0
export CHECKPOINT="none" #"/mnt/noraid/karacam/Roca/roca_log/chair_joint/model_final.pth"  # "none"
export RESUME=0 # This means from last checkpoint
export OUTPUT_DIR=/mnt/noraid/karacam/Roca/roca_log/chair_joint_icp_e2e
