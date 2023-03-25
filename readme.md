# Automatic-Label-Error-Detection
Python implementation of our paper [arXiv:2207.06104](https://arxiv.org/abs/2207.06104). This repository uses uncertainty 
quantification to find label errors in semantic segmentation datasets.

![plot](./img/0036_proposal_1.png )

## Installation 

* First, clone the repo into your home directory:

    ```
    > git clone https://github.com/mrcoee/Automatic-Label-Error-Detection.git
    ```
* Next, navigate into "~/labelerror_detection/src" and run
    
    ```
    > pip install -r requirements.txt
    ```
* And lastly also run
    
    ```
    > ./x.sh
    ```


## Usage

This model performs uncertainty quantification on the output of a neural network to find label errors. For each element 
of your dataset that you wish to evaluate it needs the input image file of the network, the corresponding logit output, 
the predicted segmentation mask, and the ground truth mask, all four in separate folders. 

```
├── data
│   ├── net_input
│   │   ├── img1.png
|   |   ├── img2.png
|   |   ├── ...
│   ├── logits
│   │   ├── img1.npy
│   │   ├── ...
│   ├── inference_output
│   │   ├── img1.png
│   │   ├── ...
|   ├── gt_masks
│   │   ├── img1.png
│   │   ├── ...
└── ...
```
This model does not automatically search for label errors in all classes of the given dataset. To define the classes of interest,
set the *CLASS_IDS* parameter in the *config.py* file

**Important:** The segmentation mask and the ground truth mask must be given as single-channel grayscale images, with each 
pixel containing a class id. The logit outputs must be saved as npy arrays in [Channel, X, Y] format.

The path to each folder and the dataset to evaluate are set in the config. Currently implemented datasets are Cityscapes, Carla, PascalVOC
and Cocostuff. If you wish to add additional datasets, you have to provide the class definition in the *label.py* file.


## Run the code
To load the data and calculate the metrics, run

```
> python3 main.py --load
```
in *.../labelerror_detection/src/*. The calculated metrics are saved in the *METRICS_DIR* defined in the config. To apply the meta seg
model on these metrics and to find label errors, run
```
> python3 main.py --evaluate
```
afterwards. The visualizations of the error proposals are saved in the *VISUALIZATIONS_DIR* folder.







**Hints**
* One part of the data you give the model is used for training and the other one for evaluation, i.e., to find label errors. You can set the 
*SPLIT_RATIO* parameter in the config file to control the amount of data used in training and evaluaton. This split is done randomly, if you want
to turn off this randomness, remove the following line from the evlaluate function in *analyze.py*
    ``` 
    np.random.shuffle(metric_fns)
    ```
* This implementation uses multiprocessing to calculate the metrics. By default, the number of workers is set to 1 in the config file. If possible, it is highly recommended to use more workers to speed up the metrics calculations

**Known Issues**

* When calculating the metric, for each component in a predicted mask we calculate the number of inner pixels divided by the number of boundary pixels. 
If a segmentation mask does not contain a single component, there are no boundary pixels and we divide by zero. Therefore it is necessary to remove all these 
to remove all these predictions before feeding the data to the model.
