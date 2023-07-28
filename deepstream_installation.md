### Tips for the installation of DeepStream 6.1 on Ubuntu 20.04
Most probably, the installation of DeepStream 6.1 on Ubuntu 20.04 will fail if following the [Nvidia Official Guide](https://docs.nvidia.com/metropolis/deepstream/6.1/dev-guide/text/DS_Quickstart.html).

The following steps worked for the installation of DeepStream 6.1 on my Ubuntu 20.04 laptop:
1. Follow the guide under 'Installing Dependencies'.
2. Do NOT install the Driver as suggested. Instead, download and install CUDA 11.6 Update 1 with the *.run* installer from [here](https://developer.nvidia.com/cuda-11-6-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local). This will also install the NVIDIA 510.47.03 Driver. (Note: the first installation attempt might fail. Reboot your machine and repeat the installation.)
3. Install CuDNN 8.4.0.27 as reported in the guide.
   `wget https://developer.nvidia.com/compute/cudnn/secure/8.4.0/local_installers/11.6/cudnn-local-repo-ubuntu2004-8.4.0.27_1.0-1_amd64.deb`
   `sudo dpkg -i cudnn-local-repo-ubuntu2004-8.4.0.27_1.0-1_amd64.deb`
   `sudo apt-get update`
   `sudo apt install libcudnn8=8.4.0.27-1+cuda11.6 libcudnn8-dev=8.4.0.27-1+cuda11.6`
5. Install TensorRT 8.2.5.1 as follows:
   `sudo apt-get install libnvinfer8=8.2.5-1+cuda11.4 libnvinfer-plugin8=8.2.5-1+cuda11.4 libnvparsers8=8.2.5-1+cuda11.4 \
   libnvonnxparsers8=8.2.5-1+cuda11.4 libnvinfer-bin=8.2.5-1+cuda11.4 libnvinfer-dev=8.2.5-1+cuda11.4 \
   libnvinfer-plugin-dev=8.2.5-1+cuda11.4 libnvparsers-dev=8.2.5-1+cuda11.4 libnvonnxparsers-dev=8.2.5-1+cuda11.4 \
   libnvinfer-samples=8.2.5-1+cuda11.4 python3-libnvinfer=8.2.5-1+cuda11.4 python3-libnvinfer-dev=8.2.5-1+cuda11.4`
   Leave *cuda11.4*. That is NOT an error. If some packages are not found, they are not needed. You can skip their installation.
5. Install librdkafka as reported in the guide.
6. Install DeepStream SDK following **Method 2** in the guide (installation from a *.tar* package). Installing with Method 1 will cause the installation of the most recent CUDA version which will cause issues with the one installed previously.
7. Install the [DeepStream_Python_Apps_Bindings_v1.1.2](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/tag/v1.1.2) with the *.whl* package.
