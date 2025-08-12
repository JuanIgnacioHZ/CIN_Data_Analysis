# Install python 3.10
#sudo pacman -S python310
#sudo yay -S python310
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10
# Create environment
python3.10 -m venv .dlc
source .dlc/bin/activate
# Install dependencies
pip install pyside6==6.4.2
pip install tensorflow
pip install tensorpack
pip install tf_slim
# Install deeplabcut
pip install deeplabcut[gui,modelzoo,wandb]==3.0.0rc9
# Launch deeplabcut
python -m deeplabcut
