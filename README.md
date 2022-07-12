# nVITA -- The adversarial example generation algorithm for time series forecasting (TSF)

This repository is for the *nVITA* algorithm.

It also contains *FGSM* and *BIM* and two baseline attacks *BRS* and *BRNV* for TSF.

*BRS* randomly selects the *sign* of the perturbation. It is the **baseline** attack method for *FGSM*, *BIM* and *FullVITA*

*BRNV* randomly selects *n* values to attack. It is the **baseline** attack method for *nVITA*

/raw_data is too large to be uploaded on Github.

## Running of the code

**NOTE** That *FGSM* and *BIM* **CANNOT** attack *RF(random forest)* while *nVITA* and *fullVITA* can.

### Step 1 - 3 data preprocessing, hyperparameter tuning, and model training

step1 to step3 has already been run

*step1_preprocessing.ipynb* **CANNOT** be run without /raw_data placed in the /data directory.

*step2_hyperparameter_tuning.ipynb* splits the data and stores them in /results/splitted_data. It also tunes the hyperparameters of the models and stores them in /experiments/metadata.json

*step3_train_model.ipynb* trains the model with the hyperparameters tunned from step2 and saves the models in /results/saved_model

### Running of the experiments

All files in /results/splitted_data and /results/saved_model are reuqired to run for step4 and step5

### Step 4 non-targeted experiments

*step4_attack_non_target.py* takes the following arguments:

**"-d", "--dfname"**, type=str, the name of dataframe, must be one of the name from **["Electricity", "NZTemp", "CNYExch", "Oil"]**

**"-s", "--seed"**, type=int, the seed integer, must be one of the seed from **["2210", "9999", "58361", "789789", "1111111"]**

**"-m", "--model"**, type=str, the attacked model name, must be one of the name from **["CNN", "LSTM", "GRU", "RF"]**

**"-a", "--attack"**, type=str, the attack name,  must be one of the attack name from **["NOATTACK", "BRS", "BRNV", "FGSM", "BIM", "NVITA", "FULLVITA"]**

**"-e", "--epsilon"**, type=float, the epsilon for the attack, must be one from **["0.05", "0.1", "0.15", "0.20"]**

**"-n", "--n"**, type=int, the *n* value for BRNV and NVITA, this will be ignored if the attack name is other than "BRNV" and "NVITA". The n value must be one of the **["1", "3", "5"]**

Optional **"--demo"**, the demo size integer, must range from 1 to 100. If we don't pass this parameter, we will run the complete experiments. If we pass an integer as demo size, the result output directory will be /examples

### Step 5 non-targeted experiments

Apart from **ALL** of the arguments *step4_attack_non_target.py* requires,

*step5_attack_target.py* takes another argument:

**"-t", "--target"**, type=str, the target direction, it must be either **"Positive"**, or **"Negative"**

## Result Interpretation

For non-targeted attacks, the **absolute error (AE)** is measured between the model prediction and ground truth y. Thus, a larger AE indicates better attack performance.

For targeted attacks, the **absolute error (AE)** is measured between the model prediction and attack goal target value. Thus, a smaller AE indicates better attack performance as we make the model prediction closer to our target after the attack.
