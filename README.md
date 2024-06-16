# IID


This is the official implementation of paper **Exploiting Inter-sample and Inter-feature Relations in Dataset Distillation (CVPR2024)** .

The repository is based on [this repo](https://github.com/VICO-UoE/DatasetCondensation) and [this repo](https://github.com/uitrbn/IDM/tree/main). Please cite their papers if you use the code. 

### Prepare pretrain models
```
cd DM+ours
python pre_model.py

```
### Basic experiments 

```
cd DM+ours
python main_DM_IID.py --dataset CIFAR10 --model ConvNet --ipc 10 --init real --dsa_strategy color_crop_cutout_flip_scale_rotate --lr_img 1 --eval_mode SS --num_exp 2 --num_eval 5

cd IDM+ours
python IDM_cifar10_IID.py --dataset CIFAR10 --model ConvNet --ipc 10 --dsa_strategy color_crop_cutout_flip_scale_rotate --init real --lr_img 0.2 --num_exp 1 --num_eval 5 --net_train_real --eval_interval 100 --outer_loop 1 --mismatch_lambda 0 --net_decay --embed_last 1000 --syn_ce --ce_weight 0.5 --train_net_num 1 --aug

```

If you use the repo, please consider citing:
```
@inproceedings{deng2024exploiting,
  title={Exploiting Inter-sample and Inter-feature Relations in Dataset Distillation},
  author={Deng, Wenxiao and Li, Wenbin and Ding, Tianyu and Wang, Lei and Zhang, Hongguang and Huang, Kuihua and Huo, Jing and Gao, Yang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17057--17066},
  year={2024}
}

```