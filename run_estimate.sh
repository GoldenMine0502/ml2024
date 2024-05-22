base_dir="."
config="config/config.yaml"
model_type=DCUNET
dataset_type=VCTK16K
epoch=50
output_data_dir=estimate/${model_type}/
chkpt_path=chkpt/${model_type}_28k/chkpt_50.pt

train=0
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


#     train_folder = '/mnt/nvme/dataset/wav16k/train/'
  #    test_folder = '/mnt/nvme/dataset/wav16k/test/'