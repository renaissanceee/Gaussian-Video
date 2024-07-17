#!/bin/bash

data_path= "/cluster/work/cvl/jiezcao/jiameng/3D-Gaussian/nerf_synthetic/hotdog/test/"

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 2000 4000 6000 8000 10000 12000 14000
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name synthetic --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000
done

