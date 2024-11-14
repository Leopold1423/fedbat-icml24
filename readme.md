# FedBAT

**FedBAT: Communication-Efficient Federated Learning via Learnable Binarization [ICML 2024]**

## Environment Setup
Please install the necessary dependencies first:
```
pip install -r requirements.txt
```

## Data Partition
Please run the following code to download and partition datasets:
```
python ./dataloader/datapartition.py 
```

## Run Experiments
Please use the scripts to run the experiments, for example:
```
./run/1.1-fmnist+fedavg.sh
./run/1.2-fmnist+fedbat.sh
```

## Citation
```
@inproceedings{li2024fedbat,
  author       = {Shiwei Li and
                  Wenchao Xu and
                  Haozhao Wang and
                  Xing Tang and
                  Yining Qi and
                  Shijie Xu and
                  Weihong Luo and
                  Yuhua Li and
                  Xiuqiang He and
                  Ruixuan Li},
  title        = {FedBAT: Communication-Efficient Federated Learning via Learnable Binarization},
  booktitle    = {Forty-first International Conference on Machine Learning, {ICML} 2024,
                  Vienna, Austria, July 21-27, 2024},
  year         = {2024},
}
```
