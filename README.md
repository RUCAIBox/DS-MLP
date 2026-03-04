# DS_MLP

This is the official Pytorch implementation for the paper: "Dual-Stream MLP is All You Need for CTR Prediction."

### Dependencies

DS-MLP has the following dependencies:

+ python 3.9+

+ pytorch 1.10+ 

Please install other required packages via `pip install -r requirements.txt`.

### Download Datasets

please download the dataset from [Criteo]([reczoo/Criteo_x1 at main](https://huggingface.co/datasets/reczoo/Criteo_x1/tree/main)), [Avazu]([reczoo/Avazu_x1 at main](https://huggingface.co/datasets/reczoo/Avazu_x1/tree/main)) and [MovieLens]([reczoo/MovielensLatest_x1 at main](https://huggingface.co/datasets/reczoo/MovielensLatest_x1/tree/main)), put them in the /data folder.

### Model Checkpoints
We provide pretrained checkpoints for both baseline reproduction models and our proposed method to facilitate reproducibility.
The checkpoints can be downloaded from:
[https://drive.google.com/file/d/1mHlKqoEs8FQjLTlzIbQi3P1fOMXkszsw/view?usp=drive_link](https://drive.google.com/file/d/1mHlKqoEs8FQjLTlzIbQi3P1fOMXkszsw/view?usp=sharing)


### Quick Start

1. cd model_zoo/DS-MLP

2. modify run_expid.py to add the FuxiCTR library to system path

   `sys.path.append('YOUR_PATH_TO_FuxiCTR/')` 

3. Create a data directory and put the  [downloaded csv files](https://github.com/reczoo/Datasets/tree/main/) in `/data/Movielens/MovielensLatest_x1`

4. Run the following script to start

#### Train the teacher model

```
 # change model_config.yaml phase:teacher_training 
 python run_expid.py --expid=DSMLP_[dataset] --gpu=0
```

#### Distillation

```
 #change model_config.yaml phase:distillation
 #add the teacher model path in DSMLP
 python run_expid.py --expid=DSMLP_[dataset] --gpu=0
```

#### Finetuning

```
# change model_config.yaml phase:finetuning
# add the student model path in DSMLP
python run_expid.py --expid=DSMLP_[dataset] --gpu=0
```

