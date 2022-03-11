#-------------------
#   CUB200
#-------------------
### ResNet50, dim = 128
## ProxyAnchor Baseline
python main.py --log_online --seed 0 --no_train_metrics --group cub200_r128_proxyanchor-baseline --project NIR \
--gpu $gpu --source_path $datapath --dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch resnet50_frozen_normalize \
--embed_dim 128 --loss oproxy --bs 90

## ProxyAnchor + NIR
python main.py --log_online --seed 0 --no_train_metrics --group cub200_r128_proxyanchor-NIR_w-0.0075 --project NIR --gpu $gpu --source_path $datapath \
--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch resnet50_frozen_normalize --embed_dim 128 --loss nir --bs 90 \
--warmup 1 --loss_nir_w_align 0.0075

## ProxyAnchor + NIR | DoublePool
python main.py --log_online --seed 0 --no_train_metrics --group cub200_r128-double_proxyanchor-NIR_w-0.0075 --project NIR --gpu $gpu --source_path $datapath \
--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch resnet50_frozen_normalize_double --embed_dim 128 --loss nir --bs 90 \
--warmup 1 --loss_nir_w_align 0.0075

### ResNet50, dim = 512
## ProxyAnchor + NIR
python main.py --log_online --seed 0 --no_train_metrics --group cub200_r512_proxyanchor-NIR_w-0.0075 --project NIR --gpu $gpu --source_path $datapath \
--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch resnet50_frozen_normalize --embed_dim 512 --loss nir --bs 90 \
--warmup 1 --loss_nir_w_align 0.0075

## ProxyAnchor + NIR | DoublePool
python main.py --log_online --seed 0 --no_train_metrics --group cub200_r512-double_proxyanchor-NIR_w-0.0075 --project NIR --gpu $gpu --source_path $datapath \
--dataset cub200 --n_epochs 100 --tau 100 --gamma 0.1 --arch resnet50_frozen_normalize_double --embed_dim 512 --loss nir --bs 90 \
--warmup 1 --loss_nir_w_align 0.0075


#-------------------
#   CARS196
#-------------------
### ResNet50, dim = 128
## ProxyAnchor Baseline
python main.py --log_online --seed 0 --no_train_metrics --group cars196_r128_proxyanchor-baseline --project NIR \
--gpu $gpu --source_path $datapath --dataset cars196 --n_epochs 100 --tau 100 --gamma 0.1 --arch resnet50_frozen_normalize \
--embed_dim 128 --loss oproxy --bs 90

## ProxyAnchor + NIR
python main.py --log_online --seed 0 --no_train_metrics --group cars196_r128_proxyanchor-NIR_w-0.01 --project NIR --gpu $gpu --source_path $datapath \
--dataset cars196 --n_epochs 100 --tau 100 --gamma 0.1 --arch resnet50_frozen_normalize --embed_dim 128 --loss nir --bs 90 \
--warmup 1 --loss_nir_w_align 0.01

## ProxyAnchor + NIR | DoublePool
# Example Schedule: --tau 144 --gamma 0.5
python main.py --log_online --seed 0 --no_train_metrics --group cars196_r128-double_proxyanchor-NIR_w-0.01 --project NIR --gpu $gpu --source_path $datapath \
--dataset cars196 --n_epochs 100 --tau 100 --gamma 0.1 --arch resnet50_frozen_normalize_double --embed_dim 128 --loss nir --bs 90 \
--warmup 1 --loss_nir_w_align 0.01

### ResNet50, dim = 512
## ProxyAnchor + NIR
# Example Schedule: --tau 34 --gamma 0.25
python main.py --log_online --seed 0 --no_train_metrics --group cars196_r512_proxyanchor-NIR_w-0.01 --project NIR --gpu $gpu --source_path $datapath \
--dataset cars196 --n_epochs 100 --tau 100 --gamma 0.1 --arch resnet50_frozen_normalize --embed_dim 512 --loss nir --bs 90 \
--warmup 1 --loss_nir_w_align 0.01

## ProxyAnchor + NIR | DoublePool
# Example Schedule: --tau 29 --gamma 0.5
python main.py --log_online --seed 0 --no_train_metrics --group cars196_r512-double_proxyanchor-NIR_w-0.01 --project NIR --gpu $gpu --source_path $datapath \
--dataset cars196 --n_epochs 100 --tau 100 --gamma 0.1 --arch resnet50_frozen_normalize_double --embed_dim 512 --loss nir --bs 90 \
--warmup 1 --loss_nir_w_align 0.01



#-------------------
#   SOP
#-------------------
### ResNet50, dim = 128
## ProxyAnchor + NIR
python main.py --log_online --no_train_metrics --seed 0 --group online_products_proxyanchor-NIR_t140-230_g02 --project NIR --gpu $gpu --source_path $datapath \
--dataset online_products --n_epochs 300 --tau 140 230 --gamma 0.2 --arch resnet50_frozen_normalize --embed_dim 128 --loss normflow --bs 90 \
--warmup 5 --loss_normflow_w_align 0.3 --loss_normflow_lrmulti 1

## ProxyAnchor + NIR | DoublePool
python main.py --log_online --no_train_metrics --seed 0 --group online_products_proxyanchor-NIR_full-double_t140-230_g02 --project NIR --gpu $gpu --source_path $datapath \
--dataset online_products --n_epochs 300 --tau 140 230 --gamma 0.2 --arch resnet50_frozen_normalize_double --embed_dim 128 --loss normflow --bs 90 \
--warmup 5 --loss_normflow_w_align 0.3 --loss_normflow_lrmulti 1

### ResNet50, dim = 512
## ProxyAnchor + NIR
python main.py --log_online --no_train_metrics --seed 0 --group online_products_proxyanchor-NIR-512_full_t140-230_g02 --project NIR --gpu $gpu --source_path $datapath \
--dataset online_products --n_epochs 300 --tau 140 230 --gamma 0.2 --arch resnet50_frozen_normalize --embed_dim 512 --loss normflow --bs 90 \
--warmup 5 --loss_normflow_w_align 0.3 --loss_normflow_lrmulti 1

## ProxyAnchor + NIR | DoublePool
python main.py --log_online --no_train_metrics --seed 0 --group online_products_proxyanchor-NIR-512_full-double_t140-230_g02 --project NIR --gpu $gpu --source_path $datapath \
--dataset online_products --n_epochs 300 --tau 140 230 --gamma 0.2 --arch resnet50_frozen_normalize_double --embed_dim 512 --loss normflow --bs 90 \
--warmup 5 --loss_normflow_w_align 0.3 --loss_normflow_lrmulti 1
