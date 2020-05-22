set -e
sudo su
cd /home/ec2-user/anaconda3/bin
pip install --upgrade pip
source activate tensorflow_p36
pip install -U tensorboard
pip install tensorflow==2.1.0-rc0
pip install seaborn --upgrade
pip install -U scikit-learn
pip install imbalanced-learn