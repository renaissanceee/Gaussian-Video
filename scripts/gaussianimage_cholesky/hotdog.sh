if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 2000 4000 6000 8000 10000 12000 14000
do
CUDA_VISIBLE_DEVICES=0 python train_video.py -d dataset/hotdog/40_frames \
--data_name synthetic --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000 --save_imgs
done
#/cluster/work/cvl/jiezcao/jiameng/3D-Gaussian/nerf_synthetic/hotdog/test/
#/cluster/work/cvl/jiezcao/jiameng/3D-Gaussian_new/benchmark_nerf_synthetic_stmt_down/hotdog/test/ours_30000/gt_1/
