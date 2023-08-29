Supporting code to the Publication [doi]

## Installation

##### Clone Repository

```bash
git clone https://github.com/Sanofi-Public/IDD-papers-fragrl.git
cd IDD-papers-fragrl
git submodule update --init  ## to get MolScore submodule
```

#### Create and activate conda environment

```bash
conda env create -f environment.yml
conda activate env_fragrl
```

## Usage

```bash
python /path/to/deepfmpo/Main.py -c configuration.txt
```

or (for DeepVL):

```bash
python /path/to/deepfmpo/DeepVL.py -c configuration.txt
```

For more details see folder [documentation](documentation).

## Contact
christoph.grebner@sanofi.com
