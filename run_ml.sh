base_dir="."
config="config/config.yaml"
model_type=DCUNET
dataset_type=VCTK16K
epoch=50
output_data_dir=estimate/${model_type}/
chkpt_path=chkpt/${model_type}/chkpt_60.pt

train=1
inference=2

python3 trainer.py \
  -b $base_dir \
  -c $config \
  -m $model_type \
  -d $dataset_type \
  -e $epoch \
  -o $output_data_dir \
  -pp $chkpt_path\
  --train $train \
  --inference $inference
