CUDA_VISIBLE_DEVICES=0
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method source_test --exp BraTs2023_GLI2PED_source_test
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_MET --TTA_method source_test --exp BraTs2023_GLI2MET_source_test
# # python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method norm --exp BraTs2023_GLI2PED_norm
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_MET --TTA_method norm --exp BraTs2023_GLI2MET_norm
# # python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method tent --exp BraTs2023_GLI2PED_tent
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_MET --TTA_method tent --exp BraTs2023_GLI2MET_tent
python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method meantimgupdate --exp BraTs2023_GLI2PED_cotta
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_MET --TTA_method cotta --exp BraTs2023_GLI2MET_cotta
# # python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method sar --exp BraTs2023_GLI2PED_sar
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_MET --TTA_method sar --exp BraTs2023_GLI2MET_sar
# # python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method sitta --exp BraTs2023_GLI2PED_sitta
# # python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_MET --TTA_method sitta --exp BraTs2023_GLI2MET_sitta
# # python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method meant --exp BraTs2023_GLI2PED_meant
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_MET --TTA_method meant --exp BraTs2023_GLI2MET_meant
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method vdptta --exp BraTs2023_GLI2PED_vdptta
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_MET --TTA_method vdptta --exp BraTs2023_GLI2MET_vdptta


# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method source_test --exp BraTs2023_GLI2PED_source_test_update
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method source_test --exp BraTs2023_GLI2SSA_source_test_update
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method sar --exp BraTs2023_GLI2PED_sar_update
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method sar --exp BraTs2023_GLI2SSA_sar_update

# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method source_test --exp BraTs2023_GLI2PED_source_test_update_estema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method source_test --exp BraTs2023_GLI2SSA_source_test_update_estema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method norm --exp BraTs2023_GLI2SSA_source_test_update_estema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method sar --exp BraTs2023_GLI2PED_sar_update_estema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method norm --exp BraTs2023_GLI2PED_norm_b10 --batch_size 10
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method sar --exp BraTs2023_GLI2SSA_sar_update_estema

# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method source_test --exp BraTs2023_GLI2PED_source_test_update_ifestema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method source_test --exp BraTs2023_GLI2SSA_source_test_update_ifestema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method source_test --exp BraTs2023_GLI2SSA_source_test_histupdate_ifestema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method sar --exp BraTs2023_GLI2PED_sar_update_estema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method sar --exp BraTs2023_GLI2SSA_sar_update_ifestema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method tent --exp BraTs2023_GLI2SSA_tent_update_ifestema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method cotta --exp BraTs2023_GLI2SSA_cotta_update_ifestema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method source_test --exp BraTs2023_GLI2SSA_source_test

# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method source_test --exp BraTs2023_GLI2PED_source_test_update_ifestema_retrain
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method source_test --exp BraTs2023_GLI2PED_source_test_hisupdate_ifestema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method source_test --exp BraTs2023_GLI2SSA_source_test_hisupdate_ifestema

# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method source_test --exp BraTs2023_GLI2PED_source_test_update_ifestema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method source_test --exp BraTs2023_GLI2SSA_source_test_update_ifestema

# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method source_test --exp BraTs2023_GLI2PED_source_test_hisupdate_estema_ifhistema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method source_test --exp BraTs2023_GLI2SSA_source_test_hisupdate_estema_ifhistema

# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method source_test --exp BraTs2023_GLI2PED_source_test_ifhisupdate_estema_ifhistema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method source_test --exp BraTs2023_GLI2SSA_source_test_ifhisupdate_estema_ifhistema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method tent --exp BraTs2023_GLI2PED_tent_ifhisupdate_estema_ifhistema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method sar --exp BraTs2023_GLI2PED_sar_ifhisupdate_estema_ifhistema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method meant --exp BraTs2023_GLI2PED_meant_ifhisupdate_estema_ifhistema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method meant --exp BraTs2023_GLI2SSA_meant_ifhisupdate_estema_ifhistema

# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method source_test --exp BraTs2023_GLI2PED_source_test_tumorifhisupdate_estema_ifhistema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method source_test --exp BraTs2023_GLI2SSA_source_test_tumorifhisupdate_estema_ifhistema

# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method source_test --exp BraTs2023_GLI2PED_source_test_clsifhisupdate_estema_ifhistema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method source_test --exp BraTs2023_GLI2SSA_source_test_clsifhisupdate_estema_ifhistema

# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method meantimgupdate --exp BraTs2023_GLI2PED_meantimgupdate_ifhisupdate_estema_ifhistema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method meantimgupdate --exp BraTs2023_GLI2SSA_meantimgupdate_ifhisupdate_estema_ifhistema

# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method meantimgupdate --exp BraTs2023_GLI2PED_meantimgupdate_ifhisupdate_estema_ifhistema_teacher_updatedX
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method meantimgupdate --exp BraTs2023_GLI2SSA_meantimgupdate_ifhisupdate_estema_ifhistema_teacher_updatedX

# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method meantimgupdate --exp BraTs2023_GLI2PED_meantimgupdate_ifhisupdate_estema_ifhistema_student_updatedX
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method meantimgupdate --exp BraTs2023_GLI2SSA_meantimgupdate_ifhisupdate_estema_ifhistema_student_updatedX

# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method meantimgupdate --exp BraTs2023_GLI2PED_meantimgupdate_estema_stu_tea_X_adaptmt_adaptloss_0.01adaptentloss
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method meantimgupdate --exp BraTs2023_GLI2PED_meantimgupdate_featupdate_2
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method sitta --exp BraTs2023_GLI2PED_sitta
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method vptta --exp BraTs2023_GLI2PED_vptta --model unet_3D_vptta
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method meantimgupdate --exp BraTs2023_GLI2PED_meantimgupdate_onlyLmt
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method meantimgupdate --exp BraTs2023_GLI2PED_meantimgupdate_onlyLre
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method meantimgupdate --exp BraTs2023_GLI2PED_meantimgupdate_ent_Lmt
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_PED --TTA_method meantimgupdate --exp BraTs2023_GLI2PED_meantimgupdate_estp90
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_MET --TTA_method meantimgupdate --exp BraTs2023_GLI2MET_meantimgupdate_featupdate
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_SSA --TTA_method meantimgupdate --exp BraTs2023_GLI2SSA_meantimgupdate_imgupdate_estema_stu_tea_X_adaptmt_adaptloss_0.01regionadaptentloss
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_MET --TTA_method meantimgupdate --exp BraTs2023_GLI2MET_meantimgupdate_estema_stu_tea_X_adaptmt_adaptloss_0.1adaptentloss

# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method source_test --exp BraTs2023_GLI2PED_source_test_clsifhisupdate_estema_ifhistema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_PED --TTA_method source_test --exp BraTs2023_GLI2PED_source_test_clsifhisupdate_estema_ifhistema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_update.py --target_domain BraTS_SSA --TTA_method source_test --exp BraTs2023_GLI2SSA_source_test_clsifhisupdate_estema_ifhistema
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method source_test --exp mms2d_A2B_source_test_b1 --batch_size 1
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method tent --exp mms2d_A2B_tent_b1 --batch_size 1
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method sitta  --exp mms2d_A2B_sitta_b1 --batch_size 1
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method norm --exp mms2d_A2B_norm_b1 --batch_size 1
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method norm --exp mms2d_A2B_norm_b2 --batch_size 2
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method sar --exp mms2d_A2B_sar_b1 --batch_size 1
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method cotta --exp mms2d_A2B_cotta_b1 --batch_size 1
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method meant --exp mms2d_A2B_meant_b1 --batch_size 1
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method meantimgupdate --exp mms2d_A2B_meantimgupdate
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method meantimgupdate --exp mms2d_A2B_meantimgupdate_histupdate
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method meantimgupdate --exp mms2d_A2B_meantimgupdate_featupdate
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method meantimgupdate --exp mms2d_A2B_meantimgupdate_bnupdate
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method meantimgupdate --exp mms2d_A2B_meantimgupdate_adaptLsem
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method meantimgupdate --exp mms2d_A2B_meantimgupdate_noSMU
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method meantimgupdate --exp mms2d_A2B_meantimgupdate_entropy
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method meantimgupdate --exp mms2d_A2B_meantimgupdate_entropy_noLmt
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method meantimgupdate --exp mms2d_A2B_meantimgupdate_fusionk_100
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method meantimgupdate --exp mms2d_A2B_meantimgupdate_estp50
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method meantimgupdate --exp mms2d_A2B_meantimgupdate_estp95
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain C --TTA_method meantimgupdate --exp mms2d_A2C_meantimgupdate_bank500
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain C --TTA_method cotta --exp mms2d_A2C_meantimgupdate_bank500
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain D --TTA_method meantimgupdate --exp mms2d_A2D_meantimgupdate_bank500
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method meantimgupdate --exp mms2d_A2B_meantimgupdate_featCon_b1 --batch_size 1
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method cotta --exp mms2d_A2B_cotta
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain B --TTA_method vptta --exp mms2d_A2B_vptta --batch_size 1 --model unet_vptta
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain C --TTA_method vptta --exp mms2d_A2C_vptta --batch_size 1 --model unet_vptta
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_2D_online_eval.py --target_domain D --TTA_method vptta --exp mms2d_A2D_vptta --batch_size 1 --model unet_vptta
# python /mnt/data1/ZhouFF/TTA4MIS/code/test_time_adaptation_3D_online_eval.py --target_domain BraTS_SSA --TTA_method source_test --exp BraTs2023_GLI2SSA_source_test