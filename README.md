# Hybrid Graph Mamba : Exploiting Non-Euclidean Geometry in Medical Image Segmentation

## Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
two GeForce RTX 3090 GPUs of 24 GB Memory.

> Note that our model also supports low memory GPU, which means you can lower the batch size


1. Configuring your environment (Prerequisites):
   
    Note that HGM is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n HGM python=3.10`.
    
    + Installing necessary packages: `pip install -r requirements.txt `.
    
2. Downloading necessary data:

    
    + downloading datasets and move it into `./data/TrainDataset/` and `./data/TestDataset/`, 
    which can be found in this [Google Drive Link (2.83G)](https://drive.google.com/file/d/1zUFsh18vlx7vX_AsdLU6NfqiOyctikVT/view?usp=drive_link). It contains:
    
    + + Polyp: It contains two Train-datasets: Kvasir-SEG (900 train samples) and CVC-ClinicDB (550 train samples); Test-datsets: CVC-300 (60 test samples), CVC-ClinicDB (62 test samples), CVC-ColonDB (380 test samples), ETIS-LaribPolypDB (196 test samples), Kvasir (100 test samples).

    + + Retinal: It contains two Train-datasets: DRIVE (20 train samples) and  CHASE DB1 (20 train samples); Test-datsets: DRIVE (20 test samples), CHASE DB1 (8 test samples).
    
    + + Automated Cardiac Diagnosis Challenge dataset (ACDC): It contains one Train-dataset: ACDC (70 train samples = 1930 axial slices); Test-datsets: ACDC (20 test samples).

    + + Synapse Multi-organ dataset: It contains one Train-dataset: Synapse (18 train samples = 2212 axial slices); Test-datsets: Synapse (12 test samples).

    + downloading pretrained weights and move it into `./HGM_ACDC_best_parameter.pth`, `./HGM_Polyp_best_parameter.pth`, `./HGM_Retinal_best_parameter.pth`, `./HGM_Synapse_best_parameter.pth`,
    which can be found in this [Google Drive Link (376.6MB)](https://drive.google.com/file/d/118OOWI2W86MfOhwi1CcRw-j8h9J8FeN3/view?usp=drive_link).
    
3. Training Configuration:

    + Assigning your costumed path, like `--train_save`, `--train_path`, `--train_save`, `--list_dir`, and `--results_save_place` in `MyTrain.py`.
    
    + Just enjoy it!

4. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `MyTest.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).
    
    + Just enjoy it!
