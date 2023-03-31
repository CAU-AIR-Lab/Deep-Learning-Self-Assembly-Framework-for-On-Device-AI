# Classification

## Requirements
- Pytorch
- Torchvision
- MySQL

## Run
```bash
python -m pip install -U pip
pip install -r requirements.txt
python db_init.py
python main.py
python sample_network.py
```

## Configuration
This program use CIFAR-10 dataset.
If you want to use another datasets, you can modify dataloaders_cifar10.py and config_cifar10.py
All setting parameters are listed in the config_cifar10.py file


