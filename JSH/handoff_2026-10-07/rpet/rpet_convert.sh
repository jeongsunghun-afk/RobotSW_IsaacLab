#!/bin/bash
R=/mnt/ssd1/jsh/rpet_handoff/leg_locomotion_handoff_20260922
U=$R/assets/urdf/Leg_URDF2/urdf/Leg_abs.urdf
OUT=$R/assets/usd_51/Leg_51.usd
source /mnt/ssd1/jsh/miniconda3/etc/profile.d/conda.sh; conda activate isaac-5.1
cd /mnt/ssd1/jsh/RobotSW_IsaacLab
export CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export PYTHONPATH=/mnt/ssd1/jsh/np126_shim:${PYTHONPATH:-}
mkdir -p $(dirname $OUT)
echo "[convert] URDF: $U"
echo "[convert] OUT : $OUT"
./isaaclab.sh -p scripts/tools/convert_urdf.py $U $OUT --headless
echo "CONVERT_EXIT=$?"
ls -la $OUT 2>/dev/null; find $(dirname $OUT) -maxdepth 2 | head -10
