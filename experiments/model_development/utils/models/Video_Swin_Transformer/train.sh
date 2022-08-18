echo "beginning preparation of data"
python3 prepare_data.py
echo "completed preparation of data"

# TODO: uncomment!
echo "beginning training"
timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
results_filename=$timestamp.pkl
python3 tools/train.py configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py --resume-from pretrained/swin_base_patch244_window877_kinetics400_22k.pth --validate --test-best --seed 22 --deterministic --out $results_filename
echo "completed training"

echo "\nall done :)"
