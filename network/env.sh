# System config
export OMP_NUM_THREADS=1
export NUM_WORKERS=4
export SEED=2021
export CUDA_VISIBLE_DEVICES=2

# NOTE: Change the data config based on your setup!
# JSON files
export DATA_DIR=/mnt/noraid/karacam/Roca/Data/Dataset/Custom5Class
# Resized images with intrinsics and poses
export IMAGE_ROOT=/mnt/noraid/karacam/Roca/Data/Images #/mnt/noraid/karacam/AlignmentNew/Resized400k/tasks/scannet_frames_25k #$HOME/Thesis/ROCA/Data/Images
# Depths and instances rendered over images
export RENDERING_ROOT=/mnt/noraid/karacam/Roca/Data/Rendering #/mnt/noraid/karacam/AlignmentNew/Resized400k/tasks/scannet_frames_25k #$HOME/Thesis/ROCA/Data/Rendering
# Scan2CAD Full Annotations
export FULL_ANNOT=/mnt/noraid/karacam/Roca/Data/Dataset/Custom5Class/full_annotations_centered.json #$HOME/Thesis/ROCA/Data/full_annotations.json

# Model configurations
export RETRIEVAL_MODE="joint" #joint+comp, resnet_resnet+image+comp, nearest
export RETRIEVAL_LOSS="triplet" #bce, triplet
export E2E=1
export NOC_WEIGHTS=1

export JOINT_MODEL_PATH="/mnt/noraid/karacam/ThesisData/joint_learning_retrieval_deformation/log"
export JOINT_WITH_ICP=1

# Train and test behavior
export EVAL_ONLY=1
export CHECKPOINT="/mnt/noraid/karacam/Roca/roca_log/joint_icp_5class/model_final.pth" #"/mnt/noraid/karacam/Roca/roca_log/chair_joint_icp_deform_reg/model_final.pth"  # "none"
export RESUME=0 # This means from last checkpoint
export OUTPUT_DIR=/mnt/noraid/karacam/Roca/roca_log/joint_icp_5class
