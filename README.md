# UTN_Captcha_Detector
This is a project to detect CAPTCHA using computer vision

Main Resource for the loss function used: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection

![Project_Structure](docs_and_results/images/project_structure.png "Project Structure")

# Contributing guidelines
1. Please check linting issues with flake8

    ```flake8```
    
# CIP Pool setup
Since the CIP pool computers are limited in memory, the project has to be setup in the /var/lit2425/<your_team>/computer_vision directory.
In order to direct the pip package manager to install packages in a separate directory:
```
mkdir pip_cache
python -m venv "captcha_env"
source captcha_env/bin/activate
pip install -r UTN_Captcha_Detector/requirements.txt --cache-dir ./pip_cache/
```

# Alex Cluster setup
First create the conda environment if it doesn't exist
```
conda env create -f environment.yml
```
Ensure main.py works in a quick interactive session

```
salloc --gres=gpu:a40:1 --time=0:30:00
conda activate captcha_env
python main.py
```

Submit the job

```
sbatch captcha.job
```

# Local Setup
activate the environment, change to desired paramters in configs/configs_common.yaml

```
python main.py configs/configs_common.yaml
```