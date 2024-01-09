ROM nvcr.io/nvidia/tensorflow:23.04-tf2-py3

RUN apt -y install wget
RUN apt -y install emacs-nox
RUN apt -y install git
RUN apt -y install llvm #llvm-devel llvm-static
RUN apt -y install xauth
RUN apt -y install patch
RUN apt -y install nodejs
RUN apt -y install openssl
RUN apt-get update
RUN apt -y install openssh-client
RUN apt -y install openssh-server
RUN apt -y install rsh-server
RUN apt -y install libboost-all-dev

RUN python3 -m pip install --upgrade numpy==1.22
RUN python3 -m pip install wheel
RUN python3 -m pip install pycuda
RUN python3 -m pip install gpustat
RUN python3 -m pip install setGPU
RUN python3 -m pip install ipython
RUN python3 -m pip install h5sparse
RUN python3 -m pip install hepunits
RUN python3 -m pip install asdf==2.4.2 # dependency issue between hepstats and pennylane thourhg semnantic-version
RUN python3 -m pip install scikit-hep
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install scikit-image
RUN python3 -m pip install scikit-optimize

RUN jupyter serverextension enable --py jupyterlab --sys-prefix
RUN python3 -m ipykernel install

RUN JUPYTER_RUNTIME_DIR=$HOME/jupyter-runtime

#packages for ROOT
RUN apt -y install dpkg-dev
RUN apt -y install cmake
RUN apt -y install g++
RUN apt -y install gcc
RUN apt -y install binutils
RUN apt -y install libx11-dev
RUN apt -y install libxpm-dev
RUN apt -y install libxft-dev
RUN apt -y install libxext-dev
RUN apt -y install python
RUN apt -y install libssl-dev
