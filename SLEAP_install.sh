# Install python 3.7
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7
# Create env
python3.7 -m venv .sleap
# Activate env
source .sleap/bin/activate
# Install SLEAP
pip install sleap[pypi]==1.4.1
# Launch SLEAP
sleap-label 
