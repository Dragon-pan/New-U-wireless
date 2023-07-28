# Adaptive Video Streaming with Real-time Object Detection using Yolo

This repository contains the code for the above project.
The following guide assumes all requirments have been installed. One such requirement is the [Deepstream SDK](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html).
Moreover, [Python bindings for Deepstream](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps) are required (follow the easy installation through the 'wheel' package). [GStreamer](https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c) is also required. It comes with Jetpack if using any Nvidia Jetson, for example. Some smaller dependencies might still be required. However, most of them will be fulfilled when configuring the PGIE following the tutorial below.

* A guide for installing DeepStream 6.1 is available [here](deepstream_installation.md)

## Running Server and Client
* Before running the Client, the Server needs to be started. This is because the feedback messaging system uses a TCP connection Server <-> Client.
* To run the Server: `sudo python3 main_server.py`. Currently the Primary GPU Inference Engine (PGIE) uses Yolo-NAS-S configured following this [tutorial](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/docs/YOLONAS.md)
* Once the Server is successfully running, start the Client with: `sudo python3 main_client.py`
* NOTE: Client and Server usually run on different devices. You will need to set Client and Server IP addresses (on the same network) accordingly in `main_client.py` and `main_server.py`


