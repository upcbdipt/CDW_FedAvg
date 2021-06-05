# CDW_FedAvg
The implementation of CDW_FedAvg in the paper "Blockchain-Based Federated Learning for Device Failure Detection in Industrial IoT"
## Getting Started
### Clone the repo
```sh
git clone https://github.com/upcbdipt/CDW_FedAvg.git && cd CDW_FedAvg
```
### To run with local or virtual environment
Install dependencies using **python 3.6+** (recommend using a virtualenv):
```sh
pip install -r requirements.txt
```
### To prepare the data
```sh
python 0_prepare_data.py
```
### To reduce dimension of training data
```sh
python 1_reduce_dimension.py
```
### To train the model
```sh
python 2_run.py
```
### To plot the results
```sh
python 3_plot.py
```
## Citation
If you use this work, please cite:
``` 
  @ARTICLE{9233457,
  author={Zhang, Weishan and Lu, Qinghua and Yu, Qiuyu and Li, Zhaotong and Liu, Yue and Lo, Sin Kit and Chen, Shiping and Xu, Xiwei and Zhu, Liming},
  journal={IEEE Internet of Things Journal}, 
  title={Blockchain-Based Federated Learning for Device Failure Detection in Industrial IoT}, 
  year={2021},
  volume={8},
  number={7},
  pages={5926-5937},
  doi={10.1109/JIOT.2020.3032544}}
```
## License 
CDW_FedAvg is distributed under [Apache 2/lic.0 license](http://www.apache.orgenses/LICENSE-2.0).

Contact: Weishan Zhang (zhangws@upc.edu.cn)