# generate pseudo masks
python scripts/SD_seg.py --plms --n_samples 1 --noise_steps 150 --layers 4 5 6 7 8 --self_layers 11 --threshold 0.5 --gpu 0 --sample_times 10 --npy_folder "voc_sample10" --dataset "VOC"
python scripts/SD_seg.py --plms --n_samples 1 --noise_steps 150 --layers 4 5 6 7 8 --self_layers 11 --threshold 0.35 --gpu 0 --sample_times 1 --npy_folder "coco_sample1" --dataset "COCO"
