
Instructions to build singularity container

1. bash build_container.sh

This script will set the TMPDIR and SINGULARITY_CACHEDIR as /data1/tmp a local tmp dir on compute-0-26. Because of user quota there is not enough space on /tmp. 

Option fakeroot is used because users need to be root to build an image. A user can be added to have fakeroot permissions by running

singularity config fakeroot --add username

see https://docs.sylabs.io/guides/3.5/user-guide/cli/singularity_config_fakeroot.html

may also need to run: echo 10000 > /proc/sys/user/max_user_namespaces

Root is used by python, and python3.9 is needed for tensorflow and other ml packages. Root is therefore installed from source because all precompiled binaries are for python3.6. Once inside the container user will need to alias python3.9 and source thisroot.sh (run script setup.sh). 

The singularity image created will be a sandbox. 

Instructions to run the container

1. singularity shell --fakeroot --nv osuML/ 

Once inside the container user should source setup.sh to alias the correct python version and source root.

The option nv is used to include nvidia in the container. Running "nvidia-smi" will confirm that the container is able to access the GPUs. As another test one can open python3.9 import tensorflow as tf and run: 

tf.config.list_physical_devices('GPU')

this will list the available GPUs. 

Note: The build.def script is based on a script from Jean-Roche Vlimant https://urldefense.com/v3/__https://github.com/cmscaltech/gpuservers/blob/master/singularity/cutting_edge.singularity__;!!KGKeukY!3XvK2R0Vy5-JHWiszOTH73SnmRqCVrqJ0mNRZoyGCJxLwmcdHBn3C2LqJE7ZU7urg_Y2rw-9QqL8bDLto7IxjewsK-atYR9sxA$ 
