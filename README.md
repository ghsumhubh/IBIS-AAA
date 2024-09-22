# IBIS AAA
Our implementation of the AAA models for the 2024 [IBIS challenge](https://ibis.autosome.org/home)

## Requirements
We tested our software on the following configuration on a Linux machine:

- Python 3.9.18
- Numpy 1.26.4
- Pandas 2.1.2
- Tensorflow 2.15.0.post1
- Keras 2.15.0
- Scipy 1.13.0
- Scikit-learn 1.3.2
- h5py 3.10.0

## Setting up the folders
To set up the folder hirarchy, first create the folder `data` and the subfolder `data/Test`. After creating the fodlers, extract the content of `IBIS.test_data.Final.v1` to `data/Test/`, specifically the `*.fasta` files for HTS, GHTS and CHS. Next, extract the `train/HTS` subfolder (not just the content as before) from `IBIS.train_data.Final.v1` to `data`. 

## Running the tool
To run the preprocessing script, use the following command:
```sh
python preprocess.py
```

After preprocessing is done, to train the models and generate predictions run:
```sh
python HTS.py
```

All models will be trained on HTS data, and produce predictions on HTS, GHTS and CHS. To view the predictions on any given experiment, navigate to `output/X/HTS/`. For example, for HTS->GHTS predictions navigate to `outout/GHTS/HTS/`.