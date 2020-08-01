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
command
```
All motions are of length 60.  

- NTU-RGBD
```sh
command
```
All motions are of length 60.  

- CMU Mocap
```sh
command
```
All motions are of length 100.  

Model files and intermediate data will be stored in `./checkpoints`

### Test and Animation
If you are generating results from models with Lie part, you need to download the corresponding datasets and place them in`/dataset`. Because our model need to sample skeletons from real human datasets.

The animation results will appear in `eval_results/`

- HumanAct12
```sh
command
```
All motions are of length 60.  

- NTU-RGBD
```sh
command
```
All motions are of length 60.  

- CMU Mocap
```sh
command
```
You could change the argument `replic_times` to get more generated motions. If you're testing the model you trained by you own, please replace the argument `name` with the name of checkpoint model you want to test.

---
#### Citation
If you find this model useful for you research, please consider citing [our work](https://ericguo5513.github.io/action-to-motion/).

#### Misc
Contact Chuan Guo at cguo2 at ualberta.ca for any questions or comments
