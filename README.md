<p align="center">
  <h3 align="center">RecGURU</h3>

<!-- ABOUT THE PROJECT -->
## About The Project
Source code and baselines for the RecGURU paper "RecGURU: Adversarial Learning of Generalized User Representations for Cross-Domain Recommendation (WSDM 2022)"

## Code Structure
```bash
RecGURU  
├── README.md                                 Read me file 
├── data_process                              Data processing methods
│   ├── __init__.py                           Package initialization file     
│   └── amazon_csv.py                         Code for processing the amazon data (in .csv format)
│   └── business_process.py                   Code for processing the collected data
│   └── item_frequency.py                     Calculate item frequency in each domain
│   └── run.sh                                Shell script to perform data processing  
├── GURU                                      Scripts for modeling, training, and testing 
│   ├── data                                  Dataloader package      
│     ├── __init__.py                         Package initialization file 
│     ├── data_loader.py                      Customized dataloaders 
│   └── tools                                 Tools such as loss function, evaluation metrics, etc.
│     ├── __init__.py                         Package initialization file
│     ├── lossfunction.py                     Customized loss functions
│     ├── metrics.py                          Evaluation metrics
│     ├── plot.py                             Plot function
│     ├── utils.py                            Other tools
│  ├── Transformer                            Transformer package
│     ├── __init__.py                         Package initialization 
│     ├── transformer.py                      transformer module
│  ├── AutoEnc4Rec.py                         Autoencoder based sequential recommender
│  ├── AutoEnc4Rec_cross.py                   Cross-domain recommender modules
│  ├── config_auto4rec.py                     Model configuration file
│  ├── gan_training.py                        Training methods of the GAN framework
│  ├── train_auto.py                          Main function for training and testing single-domain sequential recommender
│  ├── train_gan.py                           Main function for training and testing cross-domain sequential recommender
└── .gitignore                                gitignore file
```

<!-- Dataset -->
## Dataset
1. The public datasets: Amazon view dataset at: https://nijianmo.github.io/amazon/index.html
2. Collected datasets: https://drive.google.com/file/d/1NbP48emGPr80nL49oeDtPDR3R8YEfn4J/view 
3. Data processing: 
#### Amazon dataset: 
    ```shell
    cd ../data_process
    python amazon_csv.py   
    ```

#### Collected dataset
    ```shell
    cd ../data_process
    python business_process.py --rate 0.1  # portion of overlapping user = 0.1   
    ```
   After data process, for each cross-domain scenario we have a dataset folder:
   ```shell
   ."a_domain"-"b_domain"
   ├── a_only.pickle         # users in domain a only
   ├── b_only.pickle         # users in domain b only
   ├── a.pickle              # all users in domain a
   ├── b.pickle              # all users in domain b
   ├── a_b.pickle            # overlapped users of domain a and b   
   ```
   Note: see the code for processing details and make modifications accordingly.

<!-- Run -->
## Run
1. Single-domain Methods: 
    ```shell
    # SAS
    python train_auto.py --sas "True"
    # AutoRec (ours)
    python train_auto.py 
    ```
2. Cross-Domain Methods: 
    ```shell
    # RecGURU
    python train_gan.py --cross "True"
    ``` 

