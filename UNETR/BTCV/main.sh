python main.py \
    --json_list='/home/yusongli/Documents/shidaoai_new_project/data/meta_data2.yaml' \
    --gpu=0 \
    --max_epochs=100 \
    --roi_x=96 \
    --roi_y=96 \
    --roi_z=96 \
    --model_name='unetr' \
    --pos_embed='conv' \
    --trainset_cache_num=26 \
    --valset_cache_num=6 \
    --testset_cache_num=5 \
    --a_min=0 \
    --a_max=1500 \
    --workers=8 \
    --val_every=1 \
    --save_checkpoint

    # --use_normal_dataset \
