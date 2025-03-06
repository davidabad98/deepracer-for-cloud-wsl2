# **AWS DeepRacer for Cloud - GPU Setup on WSL2 (Ubuntu 20.04)**  

## **Description**  
This guide provides a step-by-step setup for running **AWS DeepRacer for Cloud** with **GPU acceleration** on **WSL2 Ubuntu 20.04** (Windows 11).  
It is intended for developers who want to leverage GPU computing inside Docker containers for DeepRacer simulations, considering that sometimes setting up Tensorflow and CUDA to be compatible in docker containers when run from WSL2 can be a headhache due to version mismatches and/or missing drivers and packages.

---

## **Table of Contents**  
- [Prerequisites](#prerequisites)  
- [Useful Links](#useful-links)  
- [Installation & Setup](#installation--setup)  
  - [1. Enable WSL2 and Install Ubuntu 20.04](#1-enable-wsl2-and-install-ubuntu-2004)  
  - [2. Install required packages](#2-install-required-packages)  
  - [3. NVIDIA Toolkit and CUDA setup](#3-nvidia-toolkit-and-cuda-setup)  
  - [4. Install GPU deepracer image to use in Docker](#4-install-gpu-deepracer-image-to-use-in-docker)  
  - [5. DRfC Info - Changes to specific files](#5-drfc-info-changes-to-specific-files)  

---

## **Prerequisites**  
Before proceeding, ensure you have:  
- **Windows 11** with **WSL2** enabled  
- **Ubuntu 20.04** running inside WSL2  
- **NVIDIA GPU** with the latest drivers installed  
- **Docker** installed in WSL2  

---

## **Useful Links**  
- [AWS DeepRacer for Cloud GitHub Repository](https://github.com/aws-deepracer-community/deepracer-for-cloud)
- [AWS DeepRacer Documentation](https://docs.aws.amazon.com/deepracer/)
- [Installing Deepracer-for-Cloud Documentation](https://aws-deepracer-community.github.io/deepracer-for-cloud/installation.html)
- [Installing Deepracer-for-Cloud with WSL2](https://aws-deepracer-community.github.io/deepracer-for-cloud/windows.html)
- [NVIDIA CUDA on WSL2 Documentation](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)  
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)  
- [Microsoft WSL2 Documentation](https://learn.microsoft.com/en-us/windows/wsl/)
- [known Issue #1](https://github.com/tensorflow/tensorflow/issues/68710)
- [known Issue #2](https://github.com/NVIDIA/nvidia-container-toolkit/issues/520)
- [Helpful article #1](https://awstip.com/deepracer-for-cloud-drfc-local-setup-3c6418b2c75a)  
- [Helpful article #2](https://medium.com/@marsmans/how-i-got-into-the-top-2-in-aws-deepracer-32127a364212)
- [Helpful article #3](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html?icmpid=docs_deepracer_console)
- [Helpful article #4](https://medium.com/@autonomousracecarclub/visualizing-aws-deepracer-waypoints-9b94e6311b7a) *

*_models that perform in the top few spots most likely utilize the continuous action space_  

---

## **Installation & Setup**  

### **1. Enable WSL2 and Install Ubuntu 20.04**  
```sh
wsl --install -d Ubuntu-20.04
```
To list the available instances:  
```sh
wsl --list --verbose
```
To change the default wsl startup instance:  
```sh
wsl --set-default Ubuntu-20.04
```
To start wsl instance (~ for root):  
```sh
wsl ~ -d Ubuntu-20.04
```
To stop wsl instance:  
```sh
wsl --shutdown Ubuntu-20.04
```

### **2. Install required packages**  
```bash
sudo apt update
sudo apt update
sudo apt-get install jq awscli python3-boto3 docker-compose
pip3 install boto3
pip3 install pyyaml
```
For dealing with video files produced from dr evaluation jobs:  
```bash
sudo apt-get install ffmpeg
sudo apt-get install jq
```

### **3. NVIDIA Toolkit and CUDA setup** 
Install and review CUDA version
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
ls /usr/local/ | grep cuda
```
```bash
nvidia-smi
nvcc --version
```

Update NVIDIA Container Toolkit
This is to fix an issue of the awsdeepracercommunity/deepracer-simapp docker container not being able to read the GPU: tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:268] failed call to cuInit: CUDA_ERROR_NOT_FOUND: named symbol not found
But this is not directly visible when running DRfC. When doing so, you will get: tensorflow/core/common_runtime/colocation_graph.cc:1213] Failed to place the graph without changing the devices of some resources. Some of the operations (that had to be colocated with resource generating operations) are not supported on the resources' devices. Current candidate devices are [
  /job:localhost/replica:0/task:0/device:CPU:0]. Which basically indicates that the GPU is not being detected.
  
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
```bash
sudo apt-get install -y nvidia-container-toolkit
nvidia-ctk --version
```
```bash
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### **4. Install missing GPU deepracer image to use in Docker** 
Check the available docker images:  
```bash
sudo service docker start
docker images
```

Pull the corresponding GPU image:  
```bash
docker pull awsdeepracercommunity/deepracer-simapp:5.3.3-gpu
docker run --rm --gpus all awsdeepracercommunity/deepracer-simapp:5.3.3-gpu nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

Check that Tensorflow is able to use GPU inside the container:  
```bash
docker run --rm --gpus all -it awsdeepracercommunity/deepracer-simapp:5.3.3-gpu bash
root@d2ea70cebb96:/opt/simapp# python3 -c "import tensorflow as tf; print(tf.__version__)"
root@d2ea70cebb96:/opt/simapp# python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### **5. DRfC Info - Changes to specific files** 
### **1. Environment Variables**  
These variables are defined in **`activate.sh`** and are essential for configuring the simulation environment.

### **2. Configuration Parameters**  

#### **System-wide Configuration (`system.env`)**  
These parameters control the system-level behavior of DeepRacer simulation:

| Variable | Description | Default Value |
|----------|------------|---------------|
| `DR_SIMAPP_VERSION` | Specifies the DeepRacer simulation container version | `5.3.3-gpu` |
| `DR_ANALYSIS_IMAGE` | Determines whether to use CPU or GPU for analysis | `gpu` |
| `DR_ROBOMAKER_MOUNT_LOGS` | Enables mounting of logs for debugging | `True` |
| `DR_DOCKER_STYLE` | Defines Docker execution style (e.g., Swarm or Standalone) | `swarm` |

---

### **3. Training-Specific Parameters (Race Track)**  

#### **Training Configuration (`run.env`)**  
These parameters affect DeepRacer training and race track simulation:

| Variable | Description | Default Value |
|----------|------------|---------------|
| `DR_EVAL_SAVE_MP4` | Enables saving of evaluation runs as MP4 videos | `True` |

---

### **4. Initial Setup Script (`init.sh`)**  

*Important IP Address update in init.sh:*  
Get message "Error response from daemon: could not choose an IP address to advertise since this system has multiple addresses on interface <your_interface> ..." when running ./bin/init.sh -c local -a cpu
It means you have multiple IP addresses and you need to specify one within ./bin/init.sh.
If you don't care which one to use, you can get the first one by running ifconfig | grep $(route | awk '/^default/ {print $8}') -a1 | grep -o -P '(?<=inet ).*(?= netmask).
Edit ./bin/init.sh and locate line docker swarm init and change it to docker swarm init --advertise-addr <your_IP>.
Rerun ./bin/init.sh -c local -a cpu
```bash
ip addr
```

Before running the simulation, execute **`init.sh`** to pull the required Docker container images.  
```bash
bin/init.sh -a gpu -c local
source bin/activate.sh
```

### **4. Copy Defaults**
```bash
cp defaults/hyperparameters.json custom_files/
cp defaults/model_metadata.json custom_files/
cp defaults/reward_function.py custom_files/
```

### **5. After changing configs in (`system.env`) or (`run.env`) **
```bash
dr-update
```

### **6. After changing the reward function **
```bash
dr-upload-custom-files
```

### **7. Train again overwriting previous training **
```bash
dr-stop-training
dr-start-training -w 
```

### **8. Start the Viewer to visualize training and evaluation in browser **
run this in a new bash:  
```bash
dr-start-viewer
dr-update-viewer
```

### **9. useful AWS s3 config **
```bash
aws s3 ls
```

## Contributing
Feel free to contribute by submitting issues or pull requests to improve this guide.
