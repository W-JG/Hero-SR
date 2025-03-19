CUDA_VISIBLE_DEVICES=0 python metric_img.py \
--hq_folder result/RealSR \
--gt_folder dataset/RealSR_CenterCrop/test_HR \
--metric_path result/result.csv \
--batch_size 1