CUDA_VISIBLE_DEVICES=0
python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method tegda --exp BraTs2023_GLI2PED_TEGDA
python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method tegda --exp MMS_A2B_TEGDA
python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain C --TTA_method tegda --exp MMS_A2C_TEGDA
python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain D --TTA_method tegda --exp MMS_A2D_TEGDA