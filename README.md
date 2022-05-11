# Lightning Prototype

### 1. Quick Start

```shell script
# clone the project 
git clone git@github.com:celsofranssa/LightningPrototype.git

# change directory to project folder
cd LightningPrototype/

# Create a new virtual environment by choosing a Python interpreter 
# and making a ./venv directory to hold it:
virtualenv -p python3 LightningPrototype/

# activate the virtual environment using a shell-specific command:
source ./LightningPrototype/bin/activate

# install dependecies
pip install -r requirements.txt

# setting python path
export PYTHONPATH=$PATHONPATH:<path-to-project-dir>/LightningPrototype/

# (if you need) to exit virtualenv later:
deactivate
```