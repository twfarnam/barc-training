

To run on new VPS, and run the following:

    sudo apt-get install python3-pip
    pip3 install -r requirements.txt
    pip3 install tensorflow-gpu==1.5.0

    curl -O
  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda-9-0

Make sure that the driver is installed by running:

    nvidia-smi

Download the NN drivers and run:

    sudo dpkg -i libcudnn7_7.0.4.31-1+cuda9.0_amd64.deb

Upload the data:

    tar -c images/ barc.db > data.tar
    scp data.tar tim@35.197.88.109:~/barc-training/data.tar

From the trainer machine:

    cd barc-trainer
    rm -rf images/ barc.db
    tar xf data.tar
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
    ./main.py -h

To run tensorboard:

    tensorboard --logdir ./log


