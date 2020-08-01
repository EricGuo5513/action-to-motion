## <b>Action2Motion: Conditioned Generation of 3D Human Motions</b> 
### [[Project Page]](https://ericguo5513.github.io/action-to-motion/)  [[Paper]](https://arxiv.org/pdf/2007.15240.pdf)<br>


There are 4 steps to run this code
* Python Virtual Environment and dependencies
* Data download and preprocessing
* Training
* Test and Animation


----
### Python Virtual Environment
Anaconda is recommended to create the virtual environment

```sh
conda create -f environment.yaml
source activate torch-action2pose
```

----
### Data & Pre-trained Models

We use three datasets and they are: `HumanAct12`, `NTU-RGBD` and `CMU Mocap`. All datasets have been properly pre-transformed to better fit our purpose. Details could are provided in our project [webpage](https://ericguo5513.github.io/action-to-motion/) and dataset documents. 

**If you just want to play our pre-trained models without Lie version, you don't need to download datasets.**

#### Download HumanAct12 Dataset
If you'd like to use HumanAct12 dataset, download the data folder [here](https://drive.google.com/drive/folders/1hGNkxI9jPbdFueUHfj8H8zC2f7DTb8NG?usp=sharing), and place it in `dataset/`

#### Download NTU-RGBD Dataset
If you'd like to use NTU-RGBD dataset, download the data folder [here](https://drive.google.com/drive/folders/1oaHZBMBne5z_ui7M1Keu3Nx1CD7f141L?usp=sharing), and place it in `dataset/`

#### Download CMU Mocap Dataset
If you'd like to use CMU-Mocap dataset, download the data folder [here](https://drive.google.com/drive/folders/1_2jbZK48Li6sm1duNJnR_eyQjVdJQDoU?usp=sharing), and place it in `dataset/`

Our pre-trained models have been involved in folder `checkpoints/`. You don't need to download them additionally.  

----
### Training
**If you just want to play our pre-trained models, you could skip this step.**
We train the models using the script `train_motion_vae.py`. All the argments and their descriptions used for training are given in `options/base_vae_option.py` and `options/train_vae_option.py`. Some of them were used during trials, but may not be used in our paper. The argments used in examples are these which produce best performances during tuning.

- HumanAct12
```sh
python train_motion_vae.py --name <Experiment_name> --dataset_type humanact12 --batch_size 128 --motion_length 60 --coarse_grained --lambda_kld 0.001 --eval_every 2000 --plot_every 50 --print_every 20 --save_every 2000 --save_latest 50 --time_counter --use_lie --gpu_id 0 --iters 50000
```
All motions are of length 60.  

- NTU-RGBD
```sh
python train_motion_vae.py --name <Experiment_name> --dataset_type ntu_rgbd_vibe  --batch_size 128 --motion_length 60 --lambda_kld 0.01 --eval_every 2000 --plot_every 50 --print_every 20 --save_every 2000 --save_latest 50 --time_counter --use_lie --gpu_id 0 --iters 50000 
```
All motions are of length 60.  

- CMU Mocap
```sh
python train_motion_vae.py --name <Experiment_name> --dataset_type mocap  --batch_size 128 --motion_length 100 --lambda_kld 0.01 --eval_every 2000 --plot_every 50 --print_every 20 --save_every 2000 --save_latest 50 --time_counter --use_lie --gpu_id 0 --iters 50000 
```
All motions are of length 100.  

Model files and intermediate data will be stored in `./checkpoints`

### Test and Animation
**If you are generating results from models with Lie part, you need to download the corresponding datasets and place them in`/dataset`.** Because our model need to sample skeletons from real human datasets.

The animation results will appear in `eval_results/`

#### Play our model with Lie

- HumanAct12
```sh
python evaluate_motion_vae.py --name vanilla_vae_lie_mse_kld001 --dataset_type humanact12 --use_lie --time_counter --motion_length 60 --coarse_grained --gpu_id 0 --replic_times 5 --name_ext _R0
```

- NTU-RGBD
```sh
python evaluate_motion_vae.py --name vanilla_vae_lie_mse_kld01 --dataset_type ntu_rgbd_vibe --use_lie --time_counter --motion_length 60 --gpu_id 0 --replic_times 5 --name_ext R0 
```

- CMU Mocap
```sh
python evaluate_motion_vae.py --name vanilla_vae_lie_mse_kld01 --dataset_type mocap --use_lie --time_counter --motion_length 60 --gpu_id 0 --replic_times 5 --name_ext R0 
```

#### Play our model without Lie

- HumanAct12
```sh
python evaluate_motion_vae.py --name vanila_vae_tf --dataset_type humanact12  --motion_length 60 --coarse_grained --gpu_id 0 --replic_times 5 --name_ext R0

```
- NTU-RGBD
```sh
python evaluate_motion_vae.py --name vanila_vae_tf_2 --dataset_type ntu-rgbd-vibe  --motion_length 60 --gpu_id 2 --replic_times 2 --name_ext R0 
```
- CMU Mocap
```sh
python evaluate_motion_vae.py --name vanila_vae_tf_2 --dataset_type mocap  --motion_length 100 --gpu_id 2 --replic_times 2 --name_ext R0 
```
You could change the argument `replic_times` to get more generated motions. If you're testing the model you trained by you own, please replace the argument `name` with the name of checkpoint model you want to test.

---
#### Citation
If you find this model or datasets useful for you research, please consider citing our work.
```sh
@inproceedings{chuan2020action2motion,  
  title={Action2Motion: Conditioned Generation of 3D Human Motions},  
  author={Chuan Guo and Xinxin Zuo and Sen Wang and Shihao Zou and Qingyao Sun and Annan Deng and Minglun Gong and Li Cheng},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia (MM '20)},  
  year={2020}  
}
```

#### Misc
Contact Chuan Guo at cguo2 at ualberta.ca for any questions or comments
