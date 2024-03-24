# DiffAttack: Evasion Attacks Against Diffusion-Based Adversarial Purification

Implementation of [DiffAttack: Evasion Attacks Against Diffusion-Based Adversarial Purification](https://arxiv.org/abs/2311.16124) [NeurIPS 2023].

Diffattack is a strong adversarial attack against diffusion-based purification defenses. We provide the following scripts for reproducing the results.

## Environment and Pretrained models

Please refer to ``requirement.txt`` for the required packages of running the codes in the repo.

Put the folder [models](https://drive.google.com/file/d/1KRfeln9t2C7kKWmi4Q0P_XDYdwTcqjFp/view?usp=sharing) and [pretrained](https://drive.google.com/file/d/1iWHGBBXix2uzoZ4zNYvu_5q91pymAkez/view?usp=sharing) under ``DiffAttack_Score_Based/`` and ``DiffAttack_DDPM_Based/``.

## Attack against score-based purification

### AdjAttack from DiffPure

Please refer to [DiffPure](https://github.com/NVlabs/DiffPure) for the adjattack against score-based diffusion purification defenses.

### DiffAttack

#### The scripts are provided in ``DiffAttack_Score_Based/run_scripts/cifar10/`` for CIFAR-10.

DiffAttack against score-based purification on CIFAR-10 with WideResNet-28-10 under Linf attack:
```commandline
sh run_cifar_rand_inf.sh SEED1 SEED2
```

DiffAttack  against score-based purification on CIFAR-10 with WideResNet-70-16 under Linf attack:

```commandline
sh run_cifar_rand_inf_70-16-dp.sh SEED1 SEED2
```

DiffAttack  against score-based purification on CIFAR-10 with WideResNet-28-10 under L2 attack:

```commandline
sh run_cifar_rand_L2.sh SEED1 SEED2
```

DiffAttack against score-based purification on CIFAR-10 with WideResNet-70-16 under L2 attack:

```commandline
sh run_cifar_rand_L2_70-16-dp.sh SEED1 SEED2
```

#### The scripts are provided in ``DiffAttack_Score_Based/run_scripts/imagenet/`` for ImageNet.

DiffAttack against score-based purification on ImageNet with ResNet-50 under Linf attack:

```commandline
sh run_in_rand_inf.sh SEED1 SEED2
```

DiffAttack against score-based purification on ImageNet with WideResNet-50-2 under Linf attack:

```commandline
sh run_in_rand_inf_50-2.sh SEED1 SEED2
```

DiffAttack against score-based purification on ImageNet with DeiT-S under Linf attack:

```commandline
sh run_in_rand_inf_deits.sh SEED1 SEED2
```

## DiffAttack against DDPM-based purification

The scripts are provided in ``DiffAttack_DDPM_Based/run_scripts/cifar10/``

### Diff-BPDA attack

Diff-BPDA attack against DDPM-based purification on CIFAR-10 with WideResNet-28-10 under Linf attack:
```commandline
sh run_cifar_ddpm_inf_bpda.sh SEED1 SEED2
```

Diff-BPDA attack against DDPM-based purification on CIFAR-10 with WideResNet-70-16 under Linf attack:

```commandline
sh run_cifar_ddpm_inf_70_bpda.sh SEED1 SEED2
```

Diff-BPDA attack against DDPM-based purification on CIFAR-10 with WideResNet-28-10 under L2 attack:

```commandline
sh run_cifar_ddpm_l2_bpda.sh SEED1 SEED2
```

Diff-BPDA attack against DDPM-based purification on CIFAR-10 with WideResNet-70-16 under L2 attack:

```commandline
sh run_cifar_ddpm_l2_70_bpda.sh SEED1 SEED2
```

### DiffAttack

DiffAttack against DDPM-based purification on CIFAR-10 with WideResNet-28-10 under Linf attack:
```commandline
sh run_cifar_ddpm_inf.sh SEED1 SEED2
```

DiffAttack against DDPM-based purification on CIFAR-10 with WideResNet-70-16 under Linf attack:

```commandline
sh run_cifar_ddpm_inf_70.sh SEED1 SEED2
```

DiffAttack against DDPM-based purification on CIFAR-10 with WideResNet-28-10 under L2 attack:

```commandline
sh run_cifar_ddpm_l2.sh SEED1 SEED2
```

DiffAttack against DDPM-based purification on CIFAR-10 with WideResNet-70-16 under L2 attack:

```commandline
sh run_cifar_ddpm_l2_70.sh SEED1 SEED2
```

### Acknowledgement

The code base is built upon [Auto-Attack](https://github.com/fra31/auto-attack) and [DiffPure](https://github.com/NVlabs/DiffPure).

If you consider our repo helpful, please consider citing:
```
@article{kang2024diffattack,
  title={DiffAttack: Evasion Attacks Against Diffusion-Based Adversarial Purification},
  author={Kang, Mintong and Song, Dawn and Li, Bo},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```