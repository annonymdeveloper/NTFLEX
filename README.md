## Installationguide for the NTFLEX framework

Follow these steps to set up the NTFLEX enviroment on your computer:

### Step 1: Clone the NTFLEX repository

First, clone the NTFLEX repository from GitHub using the following command:

```shell
git clone https://github.com/annonymdeveloper/NTFLEX.git
cd NTFLEX
```

### Step 2: Create the Conda Environment

```shell
conda create --name ntflex python=3.9.18 && conda activate ntflex
```

### Step 3: Install Dependencies

To install the required packages, run this command in the NTFLEX folder:

```shell
pip install -r requirements.txt
```

### Step 4: Preprocessing the data

The dataset created from Wikidata stores facts as quintuples with (subject, predicate/attribute, object/attribute value, since, until). Since our framework uses quadruples we need to preprocess the data. The unprocessed data can be found in data/raw. To preprocess it use following command:

```shell
python preprocessData.py
```
Already preprocessed data can be found in data/wiki:

### Step 5: Run NTFLEX Experiment

To recreate the NTFLEX results published in our paper run following command:

```shell
python train_NTFLEX.py
```

## Recreate TFLEX results

Follow these steps to recreate the TFLEX results presented in our paper

### Step 1: Cloning the repository

If not already done clone the NTFLEX repository and navigate to the TFLEX folder

```shell
git clone https://github.com/annonymdeveloper/NTFLEX.git
cd NTFLEX/TFLEX
```

### Step 2: Create the Conda Enviroment

```shell
conda create --name tflex python=3.9.18 && conda activate tflex
```

### Step 3: Install Dependencies

To install the required packages, run this command in the TFLEX folder:

```shell
pip install -r requirements.txt
```

### Step 4: Run TFLEX Experiment

To recreate the TFLEX results published in our paper run following command:

```shell
python train_TCQE_TFLEX.py --dataset "WIKI"
```

## Recreate TransEA results

Follow these steps to recreate the TransEA results presented in our paper

### Step 1: Cloning the repository

If not already done clone the NTFLEX repository and navigate to the TFLEX folder

```shell
git clone https://github.com/annonymdeveloper/NTFLEX.git
cd NTFLEX/TransEA
```

### Step 2: Create the Conda Enviroment

```shell
conda create --name transea python=3.7.16 && conda activate transea
```

### Step 3: Install Tensorflow

```shell
pip install tensorflow==1.15.0
```

### Step 4: Compile 

```shell
bash makeEA.sh
```

### Step 5: Train

To train models based on random initialization:

1. Change class Config in transEA.py and set testFlag to False

		class Config(object):
	
			def __init__(self):
				...
				self.testFlag = False
				self.loadFromData = False
				...

2.
```shell
python transEA.py
```

### Step 6: Test

To test your models:

1. Change class Config in transEA.py and set testFlag to True
	
		class Config(object):

			def __init__(self):
				...
				self.testFlag = True
				self.loadFromData = True
				...

2.
```shell
python transEA.py
```

