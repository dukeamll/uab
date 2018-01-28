Bootstrap: localimage
From: /hdd/tensorflow-latest-gpu-py3.simg

%help
This is a sigularity definition file for uab repo(https://github.com/dukeamll/uab) with Tensorflow and GPU usage.
For more info of singularity, check http://singularity.lbl.gov/quickstart.
This singularity image can help you setup environment on any linux system with necessary libraries and drivers.

%post
    echo "Installing necessary packages"
    apt-get -y update
    apt-get -y install expect
    apt-get -y install python-dev
    apt-get -y install python3-numpy
    apt-get -y install python3-dev
    apt-get -y install python3-pip
    apt-get -y install python3-wheel
    apt-get -y install libcupti-dev
    apt-get -y install zlib1g-dev
    pip install imageio
    pip install pandas
    pip install matplotlib
    pip install scipy
    pip install scikit-learn
    pip install scikit-image
    pip install six
    pip install future
    pip install pip
    pip install tqdm
    pip install pillow
    add-apt-repository ppa:ubuntugis/ppa
    apt-get -y install gdal-bin libgdal-dev
    pip install rasterio