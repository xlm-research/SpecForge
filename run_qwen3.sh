export PYTHONPATH=/mnt/lustre/yangbo1/SpecForge:$PYTHONPATH
export TORCHDYNAMO_DISABLE=1
export SGLANG_DEBUG_SKIP_WEIGHT_LOAD=1
bash /mnt/lustre/yangbo1/SpecForge/examples/run_qwen3_30b_a3b_eagle3_online.sh 
