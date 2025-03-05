# deepracer-for-cloud-wsl2
https://github.com/tensorflow/tensorflow/issues/68710
https://github.com/NVIDIA/nvidia-container-toolkit/issues/520

Guide for running DRfC in Windows WSL2 Ubuntu
AWS DeepRacer Documentation https://docs.aws.amazon.com/deepracer/
Installing Deepracer-for-Cloud Documentation: https://aws-deepracer-community.github.io/deepracer-for-cloud/installation.html
Installing on Windows: https://aws-deepracer-community.github.io/deepracer-for-cloud/windows.html
Helpful article: https://awstip.com/deepracer-for-cloud-drfc-local-setup-3c6418b2c75a
How I Got Into The Top 2% In AWS DeepRacer
https://medium.com/@marsmans/how-i-got-into-the-top-2-in-aws-deepracer-32127a364212
Input parameters of the AWS DeepRacer reward function
https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html?icmpid=docs_deepracer_console
models that perform in the top few spots most likely utilize the continuous action space
https://medium.com/@autonomousracecarclub/visualizing-aws-deepracer-waypoints-9b94e6311b7a


1 install WSL in windows
wsl useful commands:
wsl --install -d Ubuntu-20.04

wsl --list --verbose
wsl -d docker-desktop
wsl --shutdown docker-desktop


wsl --set-default docker-desktop
wsl --set-default Ubuntu
wsl -d Ubuntu
wsl ~ -d Ubuntu-20.04

2 sudo apt update
sudo apt upgrade
sudo apt-get install jq awscli python3-boto3 docker-compose
pip3 install boto3
pip3 install pyyaml


sudo apt-get install ffmpeg DEALING WITH VIDEO FILES PRODUCED FROM DR EVALUATION JOBS
sudo apt-get install jq

3 CHECK CUDA IN UBUNTU
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

nvidia-smi
nvcc --version

CHECK INSTALLED CUDA VERSIONS
ls /usr/local/ | grep cuda

UPDATE NVIDIA CONTAINER TOOLKIT 
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

THSI FIXED THE ISSUE!!!
sudo apt-get install -y nvidia-container-toolkit
nvidia-ctk --version

sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

4 INSTALL THE MISSING GPU IMAGE FOR DOCKER TO USE
CHECK AVAILABLE DOCKER IMGS
docker images


docker pull awsdeepracercommunity/deepracer-simapp:5.3.3-gpu
docker run --rm --gpus all awsdeepracercommunity/deepracer-simapp:5.3.3-gpu nvidia-smi

CHECK TENSORFLOW VERSION INSIDE THE CONTAINER
docker run --rm --gpus all -it awsdeepracercommunity/deepracer-simapp:5.3.3-gpu bash
root@d2ea70cebb96:/opt/simapp# python3 -c "import tensorflow as tf; print(tf.__version__)"
root@d2ea70cebb96:/opt/simapp# python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"


sudo service docker start
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi


DRfC repo:
ENV VARIABLES
activate.sh

CONFIG PARAMS
system.env
DR_SIMAPP_VERSION=5.3.3-gpu
DR_ANALYSIS_IMAGE=gpu
DR_ROBOMAKER_MOUNT_LOGS=True
DR_DOCKER_STYLE=swarm

PARAMS SPECIFIC TO TRAINING (RACE TRACK)
run.env
DR_EVAL_SAVE_MP4=True


init.sh INTIAL SETUP (PULL CONTAINER IMGS)

UPDATE IP IN init.sh:
ip addr
http://172.30.67.131:8080
172.30.67.131/20
Get message "Error response from daemon: could not choose an IP address to advertise since this system has multiple addresses on interface <your_interface> ..." when running ./bin/init.sh -c local -a cpu
It means you have multiple IP addresses and you need to specify one within ./bin/init.sh.
If you don't care which one to use, you can get the first one by running ifconfig | grep $(route | awk '/^default/ {print $8}') -a1 | grep -o -P '(?<=inet ).*(?= netmask).
Edit ./bin/init.sh and locate line docker swarm init and change it to docker swarm init --advertise-addr <your_IP>.
Rerun ./bin/init.sh -c local -a cpu

RUN ALWAYS FROM DRfC DIR:
bin/init.sh -a gpu -c local
source bin/activate.sh

COPY DEFAULTS
cp defaults/hyperparameters.json custom_files/
cp defaults/model_metadata.json custom_files/
cp defaults/reward_function.py custom_files/

AFTER CHANGING CONFIGS IN SYSTEM.ENV OR RUN.ENV
dr-update

AFTER CHANGING REWARD FUNCTION
dr-upload-custom-files

AWS S3 CONFIG
aws s3 ls

TRAIN AGAIN OVERWRTTING PREVIOUS TRAINING
dr-start-training -w 
dr-stop-training

dr-start-viewer
dr-update-viewer
