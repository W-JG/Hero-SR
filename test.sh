python test.py \
--sd_model_name_or_path sd-turbo \
--pretrained_model_path checkpoint/hero_sr_model.pkl \
--input_dir dataset/RealSR_CenterCrop/test_LR \
--output_dir result/RealSR \
--align_method adain \
--upscale_factor 4