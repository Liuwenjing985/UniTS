model_name=UniTS 
exp_name=UniTS_supervised_x64 
wandb_mode=online 
project_name=supervised_learning 

random_port=$((RANDOM % 9000 + 1000))
ckpt_path=newcheckpoints/UniTS_finetune_few_shot_newdata_pct05_test.pth

# Supervised learning
torchrun --nnodes 1 --nproc-per-node=1 --master_port $random_port run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model 64 \
  --des 'Exp' \
  --learning_rate 1e-4 \
  --weight_decay 5e-6 \
  --train_epochs 20 \
  --batch_size 32 \
  --acc_it 32 \
  --debug $wandb_mode \
  --project_name $project_name \
  --clip_grad 100 \
  --pretrained_weight $ckpt_path \
  --task_data_config_path data_provider/customdata_classification.yaml  


# Prompt tuning
# torchrun --nnodes 1 --master_port $random_port run.py \
#   --is_training 1 \
#   --model_id $exp_name \
#   --model $model_name \
#   --lradj prompt_tuning \
#   --prompt_num 10 \
#   --patch_len 16 \
#   --stride 16 \
#   --e_layers 3 \
#   --d_model $d_model \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 3e-3 \
#   --weight_decay 0 \
#   --prompt_tune_epoch 2 \ # Number of epochs for prompt tuning
#   --train_epochs 0 \
#   --acc_it 32 \
#   --debug $wandb_mode \
#   --project_name $ptune_name \
#   --clip_grad 100 \
#   --pretrained_weight ckpt_path.pth \ # Path of pretrained ckpt, you must add it for prompt learning 
#   --task_data_config_path  data_provider/customdata_classification.yaml # Important: Change to your_own_data_config.yaml
