# Experiments

This document specifies how to reproduce our experimental results for each dataset considered in the paper.
One can modify `CUDA_VISIBLE_DEVICES` to further specify GPU number(s) to work on. 
In case of running multiple scripts of `train_gan.py` in parallel, make sure that there is no conflict in
port allocation by specifying different `--port=$PORT` for each of these scripts.

### CIFAR-10
```
# G: SNDCGAN / D: SNDCGAN 
python train_gan.py configs/gan/cifar10/c10_b64.gin sndcgan --mode=std
python train_gan.py configs/gan/cifar10/c10_b64.gin sndcgan --mode=std --penalty=cr --aug=hfrt
python train_gan.py configs/gan/cifar10/c10_b64.gin sndcgan --mode=std --penalty=bcr --aug=hfrt
python train_gan.py configs/gan/diffaug/c10_diffaug.gin sndcgan --mode=aug_both --aug=diffaug
python train_gan.py configs/gan/cifar10/c10_b512.gin sndcgan --mode=contrad --aug=simclr --use_warmup

# G: SNDCGAN / D: SNResNet-18
python train_gan.py configs/gan/cifar10/c10_b64.gin snresnet18 --mode=std
python train_gan.py configs/gan/cifar10/c10_b64.gin snresnet18 --mode=std --penalty=cr --aug=hfrt
python train_gan.py configs/gan/cifar10/c10_b64.gin snresnet18 --mode=std --penalty=bcr --aug=hfrt
python train_gan.py configs/gan/diffaug/c10_diffaug.gin snresnet18 --mode=aug_both --aug=diffaug
python train_gan.py configs/gan/cifar10/c10_b512.gin snresnet18 --mode=contrad --aug=simclr --use_warmup

# G: StyleGAN2 / D: StyleGAN2
python train_stylegan2.py configs/gan/stylegan2/c10_style32.gin stylegan2 --mode=std \
--lbd_r1=0.1 --no_lazy --halflife_k=1000
python train_stylegan2.py configs/gan/stylegan2/c10_style64.gin stylegan2 --mode=contrad --aug=simclr \
--lbd_r1=0.1 --no_lazy --halflife_k=1000 --use_warmup
```

### CIFAR-100
```
# G: SNDCGAN / D: SNDCGAN 
python train_gan.py configs/gan/cifar100/c100_b64.gin sndcgan --mode=std
python train_gan.py configs/gan/cifar100/c100_b64.gin sndcgan --mode=std --penalty=cr --aug=hfrt
python train_gan.py configs/gan/cifar100/c100_b64.gin sndcgan --mode=std --penalty=bcr --aug=hfrt
python train_gan.py configs/gan/diffaug/c100_diffaug.gin sndcgan --mode=aug_both --aug=diffaug
python train_gan.py configs/gan/cifar100/c100_b512.gin sndcgan --mode=contrad --aug=simclr --use_warmup

# G: SNDCGAN / D: SNResNet-18
python train_gan.py configs/gan/cifar100/c100_b64.gin snresnet18 --mode=std
python train_gan.py configs/gan/cifar100/c100_b64.gin snresnet18 --mode=std --penalty=cr --aug=hfrt
python train_gan.py configs/gan/cifar100/c100_b64.gin snresnet18 --mode=std --penalty=bcr --aug=hfrt
python train_gan.py configs/gan/diffaug/c100_diffaug.gin snresnet18 --mode=aug_both --aug=diffaug
python train_gan.py configs/gan/cifar100/c100_b512.gin snresnet18 --mode=contrad --aug=simclr --use_warmup

# G: StyleGAN2 / D: StyleGAN2
python train_stylegan2.py configs/gan/stylegan2/c100_style32.gin stylegan2 --mode=std \
--lbd_r1=0.1 --no_lazy --halflife_k=1000
python train_stylegan2.py configs/gan/stylegan2/c100_style64.gin stylegan2 --mode=contrad --aug=simclr \
--lbd_r1=0.1 --no_lazy --halflife_k=1000 --use_warmup
```

### CelebA-HQ-128
```
# Default training
python train_gan.py configs/gan/celeba128/celeba128.gin sndcgan --mode=std

# Hinge loss
python train_gan.py configs/gan/celeba128/celeba128_hinge.gin sndcgan --mode=std

# Consistency regularization (CR)
python train_gan.py configs/gan/celeba128/celeba128.gin sndcgan --mode=std --penalty=cr --aug=hfrt

# Balanced CR (bCR)
python train_gan.py configs/gan/celeba128/celeba128.gin sndcgan --mode=std --penalty=bcr --aug=hfrt

# ContraD (ours)
python train_gan.py configs/gan/celeba128/celeba128.gin sndcgan --mode=contrad --aug=simclr_hq --use_warmup
```

### AFHQ datasets
```
# AFHQ-Dog: StyleGAN2
python train_stylegan2.py configs/gan/stylegan2/afhq_dog_style64.gin stylegan2_512 \
--mode=std --lbd_r1=0.5 --halflife_k=20 --evaluate_every=5000 --n_eval_avg=1 --no_gif

# AFHQ-Dog: StyleGAN2 + ContraD (ours)
python train_stylegan2_contraD.py configs/gan/stylegan2/afhq_dog_style64.gin stylegan2_512 \
--mode=contrad --aug=simclr_hq --lbd_r1=0.5 --halflife_k=20 --use_warmup \
--evaluate_every=5000 --n_eval_avg=1 --no_gif

# AFHQ-Cat: StyleGAN2
python train_stylegan2.py configs/gan/stylegan2/afhq_cat_style64.gin stylegan2_512 \
--mode=std --lbd_r1=0.5 --halflife_k=20 --evaluate_every=5000 --n_eval_avg=1 --no_gif

# AFHQ-Cat: StyleGAN2 + ContraD (ours)
python train_stylegan2_contraD.py configs/gan/stylegan2/afhq_cat_style64.gin stylegan2_512 \
--mode=contrad --aug=simclr_hq_cutout --lbd_r1=0.5 --halflife_k=20 --use_warmup \
--evaluate_every=5000 --n_eval_avg=1 --no_gif

# AFHQ-Wild: StyleGAN2
python train_stylegan2.py configs/gan/stylegan2/afhq_wild_style64.gin stylegan2_512 \
--mode=std --lbd_r1=0.5 --halflife_k=20 --evaluate_every=5000 --n_eval_avg=1 --no_gif

# AFHQ-Wild: StyleGAN2 + ContraD (ours)
python train_stylegan2_contraD.py configs/gan/stylegan2/afhq_wild_style64.gin stylegan2_512 \
--mode=contrad --aug=simclr_hq_cutout --lbd_r1=0.5 --halflife_k=20 --use_warmup \
--evaluate_every=5000 --n_eval_avg=1 --no_gif
```