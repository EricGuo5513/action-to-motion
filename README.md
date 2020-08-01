## <b>Action2Motion: Conditioned Generation of 3D Human Motions</b> 
### [[Project Page]](https://ericguo5513.github.io/action-to-motion/)  [[Paper]](https://arxiv.org/pdf/2007.15240.pdf)<br>


There are 4 steps to run this code
* Python Virtual Environment and dependencies
* Data download and preprocessing
* Training
* Evaluation

----

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
If you just want to test our model without Lie version, you don't need to download datasets.

#### Download HumanAct12 Dataset
If you'd like to use HumanAct12 dataset, download the data folder [here](https://drive.google.com/drive/folders/1hGNkxI9jPbdFueUHfj8H8zC2f7DTb8NG?usp=sharing), and place it in `dataset/`

#### Download NTU-RGBD Dataset
If you'd like to use NTU-RGBD dataset, download the data folder [here](https://drive.google.com/drive/folders/1oaHZBMBne5z_ui7M1Keu3Nx1CD7f141L?usp=sharing), and place it in `dataset/`

#### Download CMU Mocap Dataset
If you'd like to use CMU-Mocap dataset, download the data folder [here](https://drive.google.com/drive/folders/1_2jbZK48Li6sm1duNJnR_eyQjVdJQDoU?usp=sharing), and place it in `dataset/`

Our pre-trained models have been involved in folder `checkpoints/`. You don't need to download additionally.  
