

To run on new VPS, and run the following:

    sudo apt-get install python-pip
    pip install -r requirements.txt

    curl -O
  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda=8.0.61-1

Download the NN drivers and run:

    sudo dpkg -i libcudnn5_5.1.10-1+cuda8.0_ppc64el.deb

Upload the data using SCP and run the following:

    python train.py

