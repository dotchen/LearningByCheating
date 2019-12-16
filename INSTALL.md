# Setup

## Install CARLA
- Download the [released 0.9.6 binary](http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz) and use our compiled `.egg` [file](https://drive.google.com/drive/u/2/folders/1_DOO34PLQl7x7GlnEoShF17DNM6yrBp_), if you are using Python 2.7 or 3.5. You still need to download the updated Navmesh.
- Alternatively, you can compile carla from source. Clone CARLA 0.9.6 with our pedestrian fix at: https://github.com/dianchen96/carla/tree/0.9.6-lbc. Follow the instructions to compile and download the assets.

## Install our custom Navmesh
Download the modified Navmesh for Town1 and Town2: 

`Town01`: http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town01.bin

`Town02`: http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town02.bin

## Setup LBC
- Clone this repo and replace all the files inside the CARLA folder
- Install the dependencies or `conda install -f environment.yml`.
- (Optionally) Download the model checkpoints specified in [README](..), or train the models.
