Sugar Crystal Analysis Platform Documentation
 
Tensorflow 2.5.0 Setup	2
Basic Setup Requirements for setting up a full Tensorflow (desktop environment).	3
Basic Setup Requirements for setting up a Tensorflow Lite build for Raspberry Pi	3
Setup requirements for training a dataset using a desktop computer.	4
Simple Tensorflow script to open and display images using Pillow	4
Workflow	5
Raspberry Pi Image / Boot Disk / Deployment	6
Writing the image to an SD card	7
Backing up the raspberry pi image to a file (Windows).	7
Link to Sugar Crystal Analysis image (first version, no application)	7
Login information for the image	7
Developer Information	7
Pyenv	8
Gdown	8
Bazel	8
Tensorflow	8
Sugar Crystal Analysis	9
References	10
 
Tensorflow 2.5.0 Setup

Basic Setup Requirements for setting up a full Tensorflow (desktop environment).

Hardware Requirements:
-	Nvidia GPU with Pascal architecture
Software Requirements:
-	Ubuntu 16.04 and above (20.04 used in this project).
-	Python 3.7 (specifically 3.7.13 is used for this project).
-	GCC-7
-	G++-7
-	Bazel 3.7.2
-	CUDA 11.2 https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
-	CUDNN 8.1 https://developer.nvidia.com/rdp/cudnn-archive
-	Git (any version)
-	Numpy 1.19.2 (note that it states 1.19.2 is required however Tensorflow will not actually import without version 1.21.6)
-	Keras Preprocessing-1.1.2
-	Opt-Einsum 3.3.0
-	Packaging-21.3
-	Pyparsing-3.0.9
-	Wheel.0.37.1
-	Pip version 19.0 or higher (used for installing tensorflow from .whl file).
-	Technical documentation lists Ubuntu 16.04 or higher, although it is also compatible with Raspberry Pi’s Raspbian operating system, as it is a Debian based Linux distribution like Ubuntu. Ubuntu 20.04 is used for this project.

Basic Setup Requirements for setting up a Tensorflow Lite build for Raspberry Pi

Installation Requirements:
-	Ubuntu packages are installed with the commands: apt update and apt upgrade
-	As Raspberry Pi uses an ARM architecture processor (specifically the 32-bit arm7vl), and the current compatible version of Tensorflow for the 32-bit arm7vl architecture can only use Python 3.7, the pyenv package is chosen to install Python 3.7 for version control
-	The version of Tensorflow currently used for this project is Tensorflow 2.5.0, as it is the only currently compatible version of Tensorflow that can be used with raspberry pi architecture.
-	A virtual environment is created with Python used for running Tensorflow, and to avoid any potential version conflicts.
-	The python numpy 1.21.6 library is additionally used to assist with Tensorflow, but is installed alongside Tensorflow in one of the below links.
-	Python library matplotlib version 3.5.3
-	Python library PIL 9.2.0 (pillow) used for processing images on the pi.
-	Gdown is used to retrieve the wheel file that Tensorflow is installed from.

Links:
-	Tensorflow: https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/main/previous_versions/download_tensorflow-2.5.0-cp37-none-linux_armv7l_numpy1195.sh

Setup requirements for training a dataset using a Windows 10 desktop computer.

The Raspberry Pi specifications are insufficient to train the dataset; a workstation with modern processing power is required. In this case I’ve used my own personal computer with an Nvidia GTX 1080 to train the dataset. A CPU can also be used, but as Tensorflow was designed with an Nvidia GPU in mind and is stated to be 50% faster than a CPU for Tensorflow operations, I’ve chosen to use the GPU training method (starting from Pascal microarchitecture, more information on that here: https://www.nvidia.com/en-au/data-center/gpu-accelerated-applications/tensorflow/).
In addition, the developer environment used here is the Linux distribution Ubuntu 20.04 LTS. More specifically, Ubuntu is used on Windows 10 under Windows Subsystem for Linux 2 (WSL2).
This guide is used for the initial setup of Tensorflow 2.5.0 on a desktop: https://www.tensorflow.org/install/source#setup_for_linux_and_macos

Software Requirements:
-	Python 3.7-3.9
-	Pip 19.0 or higher (used after assembling Tensorflow from repository).
-	Ubuntu 16.04 (64-bit) or higher
 
CUDA and CUDnn
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

sudo apt-get update

sudo apt-get --yes install cuda-toolkit-11-2

sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-machine-learning.list'

sudo apt-get update

sudo apt-get install --yes --no-install-recommends cuda-11-2 libcudnn8=8.1.0.77-1+cuda11.2 libcudnn8-dev=8.1.0.77-1+cuda11.2

Sample test files can be found by changing directory to:

cd /usr/local/cuda-11.2/samples

Navigate to any of these directories and create a make file e.g. and test the installation:

cd /1_Utilities/ UnifiedMemoryPerf

sudo make

./UnifiedMemoryPerf

Note that the message “Your kernel may have been built without NUMA support.” Can safely be ignored as it is a result of running in a WSL2 environment.

Additional information can be found here: https://docs.nvidia.com/cuda/archive/11.2.0/wsl-user-guide/index.html
 
Simple Tensorflow script to open and display images using Pillow
                               
 #!/home/crystalmate/.pyenv/shims/python3

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
 
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_i>
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=T>
data_dir = pathlib.Path(data_dir)
 
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
 
roses = list(data_dir.glob('roses/*'))
with PIL.Image.open(str(roses[0])) as im:
        im.rotate(45).show()

Sourced from: https://www.tensorflow.org/tutorials/images/classification



 
Workflow
1.	Examine and understand data
Construct the dataset using provided images. Using labelimg to label all sugar crystals in preparation for object detection.
2.	Build an input pipeline

3.	Build a model
For now using official models: https://github.com/tensorflow/models/tree/r2.5.0/official
4.	Train the model

5.	Test the model

6.	Improve the model and repeat the process



 
Raspberry Pi Image / Boot Disk / Deployment
Writing the image to an SD card


Backing up the raspberry pi image to a file (Windows).
Requirements:
1.	Raspberry Pi’s SD card
2.	USB with SD card reader
3.	Win32 disk imager application
https://win32diskimager.download/


Link to Sugar Crystal Analysis image (first version, no application)

This simply includes a basic setup of Tensorflow Lite for the Raspberry Pi
https://drive.google.com/file/d/1ak5RcAEVTHS-esUyaS2aj0WGMtpAKLex/view?usp=sharing
Login information for the image
Username: crystalmate
Password: goldeneye

 
Developer Information
Pyenv (OUTDATED)
Source env/bin/activate used to initialise the virtual environment 
Pyenv shell 3.7.13 Used to load Python 3.7.13 for as long as the current terminal session is running.
Pyenv local 3.7.13 Used to load Python 3.7.13 from the current directory.
Pyenv global 3.7.13 Used to load Python 3.7.13 globally.
Deactivate Exits the virtual environment
eval "$(pyenv init -)" Used to initialise the virtualised version of Python, this can be added to ~/.bashrc with chosen text editor.
The following is also added to ~/.bashrc for shell startup:
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
Better to use Conda:
1.	Install conda-forge channel
2.	Set up a directory
3.	Create a new conda environment using conda activate –-name <name> python=3.7
4.	Activate using conda activate <name>
5.	The terminal output should be preceded by the name of the environment.

Docker

Information pulled from: 
https://www.tensorflow.org/install/docker
https://docs.docker.com/engine/reference/commandline/docker/
Current base image: https://hub.docker.com/layers/tensorflow/tensorflow/2.5.0-gpu/images/sha256-0cb24474909c8ef0a3772c64a0fd1cf4e5ff2b806d39fd36abf716d6ea7eefb3?context=explore
 
Useful Commands: 
Command	Description
docker pull image_name:tag	Retrieves a docker image from docker hub. E.g. usage: docker pull tensorflow/tensorflow:2.5.0-gpu
Docker image ls	Lists all docker images stored locally.
Docker run --gpus all -it --user 0 image_name:tag /bin/bash	Creates and runs a container for the image and opens a bash terminal to the container. -gpus option necessary for running Tensorflow with GPU.
Docker ps	Shows all actively running containers
Docker ps -a	Shows all containers, including stopped containers.
docker exec -it container_name /bin/bash	Opens a terminal shell to the container
Gdown
Gdown [GOOGLE_DRIVE_FILE_ID] used to retrieve large google drive files.

Bazel

By default will use –march=native for GCC to optimise the build process depending upon CPU architecture.

Bazel enjoys eating up the host memory to such a degree that it will cause the entire build process to fail as oom_reaper will kill one of the critical services for the process; therefore it’s necessary to set resource limitations using the below command:
bazel build --local_ram_resources=HOST_RAM*.75 --local_cpu_resources=HOST_CPUS*.5 //tensorflow/tools/pip_package:build_pip_package
Xserver

Windows does not natively support xserver, which is a process that Linux uses to display images. If intending to use WSL2 as a development environment, an alternative application will need to be installed that allows the WSL2 OS to display images from any python scripts.
Link: https://sourceforge.net/projects/vcxsrv/
From WSL2, .bashrc will need to include the following line:
export DISPLAY="$(grep nameserver /etc/resolv.conf | sed 's/nameserver //'):0"
This will allow the WSL2 instance upon start to display images to the xserver hosted in the Windows environment.


See: --local_cpu_resources, --local_ram_resources
https://bazel.build/reference/command-line-reference

Also add this to your /.profile

# set DISPLAY to use X terminal in WSL
# in WSL2 the localhost and network interfaces are not the same than windows
if grep -q WSL2 /proc/version; then
    # execute route.exe in the windows to determine its IP address
    DISPLAY=$(route.exe print | grep 0.0.0.0 | head -1 | awk '{print $4}'):0.0

else
    # In WSL1 the DISPLAY can be the localhost address
    if grep -q icrosoft /proc/version; then
        DISPLAY=127.0.0.1:0.0
    fi

fi

 
Tensorflow

Testing to make sure Tensorflow is installed correctly, run python from the command line:
Import tensorflow
Tensorflow.__version__
There should be no warnings or errors.
Additional note: Some documentation for python usage will specify that you need to import keras separately, however in Tensorflow 2.5.0 and above it is automatically included as part of the Tensorflow library and does not need to be specifically imported.
GPU test:
"(tf.reduce_sum(tf.random.normal([1000, 1000])))"

Disabling misc messages (typically these are a result of running Tensorflow within a virtualised environment and can usually be ignored).
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}

Additional information found here: https://www.tensorflow.org/versions/r2.5/api_docs/python/tf/compat/v1/logging
 
Sugar Crystal Analysis


 
References
Install TensorFlow with pip (2022). Tensorflow. https://www.tensorflow.org/install/pip
https://github.com/PINTO0309/Tensorflow-bin/tree/main/previous_versions
https://github.com/pyenv/pyenv
https://github.com/wkentaro/gdown
https://github.com/PINTO0309/Tensorflow-bin/blob/main/previous_versions/download_tensorflow-2.5.0-cp37-none-linux_armv7l_numpy1195.sh
https://www.virtualbox.org/manual/ch08.html#vboxmanage-convertfromraw
Author, A. A. (Year). Title. Site Name. 
https://www.tensorflow.org/install/pip
https://www.tensorflow.org/tutorials/images/classification
https://www.nvidia.com/en-au/data-center/gpu-accelerated-applications/tensorflow/

