# Scalable Transformer for High Dimensional Multivariate Time Series Forecasting
This paper was accepted by CIKM 2024. We provide the open-source code here.
This is the official repository for "Scalable Transformer for High Dimensional Multivariate Time Series Forecasting" ***(Accepted by CIKM-24)*** [[Paper]](https://dl.acm.org/doi/10.1145/3627673.3679757) <br>

🌟 If you find this work helpful, please consider to star this repository and cite our research:
```
@inproceedings{10.1145/3627673.3679757,
  author = {Zhou, Xin and Wang, Weiqing and Buntine, Wray and Qu, Shilin and Sriramulu, Abishek and Tan, Weicong and Bergmeir, Christoph},
  title = {Scalable Transformer for High Dimensional Multivariate Time Series Forecasting},
  year = {2024},
  isbn = {9798400704369},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3627673.3679757},
  doi = {10.1145/3627673.3679757},
  booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages = {3515–3526},
  numpages = {12},
  keywords = {forecasting accuracy, high-dimensional time series, multivariate time series forecasting},
  location = {Boise, ID, USA},
  series = {CIKM '24}
}
```

## Datasets
Please access the well-pre-processed Crime-Chicago and Wiki-People datasets from [[Google Drive]](https://drive.google.com/drive/folders/1O-LcxA3TGTFMpCAybA6OmRXEfjdA8q8W?usp=drive_link), then place the downloaded contents under the corresponding folders of `/dataset`

## Quick Demo
1. Clone this repository
```
git clone git@github.com:xinzzzhou/ScalableTransformer4HighDimensionMTSF.git
cd ScalableTransformer4HighDimensionMTSF
```
2. Config environment
```
conda create --name sthd python=3.9
conda activate sthd
pip install -r requirement.txt
```
3. Download datasets and place them under the corresponding folders of `/dataset`
4. Train and test the model. We provide two main.py files for demonstration purpose under the root folder. For example, you can train and test Crime-Chicago dataset by:
   
***Relation sparsity.***
Run datasets/top-k-train corr-compute.py to get the correlation, modeling with the accelerated computation - DeepGraph. 
```
python datasets/top-k-train/corr-compute.py
```

***Run the main file.***
Config the parameters and run run.py to train and evaluate the model. 
```
python run_crime.py
```


## Acknowledgement
Our implementation adapts [Time-Series-Library](https://github.com/thuml/Time-Series-Library) as the code base and have extensively modified it to our purposes. We thank the authors for sharing their implementations and related resources.
