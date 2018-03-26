

To run on new VPS, and run the following:

    sudo apt-get install python-pip
    pip install -r requirements.txt
    pip install tensorflow-gpu==1.5.0

    curl -O
  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda-8-0

N.B. the above stopped working with Linux kernel 4.13. Check the install logs carefully and if building the kernel module failed, you can follow instructions here:

https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07#install-nvidia-graphics-driver-via-runfile

https://stackoverflow.com/questions/48220265/cannot-install-nvidia-driver-in-funcion-block-cpu-fault-locked-error-implic

Make sure that the driver is installed by running:

    nvidia-smi

Download the NN drivers and run:

    sudo dpkg -i libcudnn5_5.1.10-1+cuda8.0_ppc64el.deb

Upload the data:

    tar -c data/ > data.tar
    scp data.tar tim@35.197.88.109:~/barc-training/data.tar

From the trainer machine:

    cd barc-trainer
    rm -rf data
    tar xf data.tar
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
    ./train_mobilenets.py

