# UTN_Captcha_Detector
This is a project to detect CAPTCHA using computer vision

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
# Run the code 
```
python main.py configs/configs_common.yaml
```