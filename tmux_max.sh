#!/bin/bash

# 定义split与device的映射关系
declare -A splits
splits[1]="Dermoscopy_Skin MRI_Alzheimer"
splits[2]="MRI_Brain Fundus_Retina"
splits[3]="Mamography_Breast OCT_Retina"
splits[4]="CT_Chest CT_Heart"
splits[5]="CT_Brain Xray_Chest"
splits[6]="Xray_Skeleton Xray_Dental"
splits[7]="Endoscopy_Gastroent Ultrasound_Baby"
splits[8]="Ultrasound_Breast Ultrasound_Carotid"
splits[9]="Ultrasound_Ovary Ultrasound_Brain"

# 创建tmux session
tmux new-session -d -s jailbreak
tmux send-keys "watch -n 0.1 nvidia-smi" C-m

# 遍历splits映射并创建相应的window
for key in "${!splits[@]}"; do
    read -ra ADDR <<< "${splits[$key]}"
    for split in "${ADDR[@]}"; do
        tmux new-window -n "$split"
        tmux send-keys "conda activate llava-med && python run_jailbreak.py --split $split --device $key --attack-mode gcg && python run_jailbreak.py --split $split --device $key --attack-mode pgd && python run_jailbreak.py --split $split --device $key --attack-mode mcm" C-m
    done
done


# 切换到nvidia-smi监控的窗口 附加到session
tmux select-window -t 0
tmux attach-session -t jailbreak
