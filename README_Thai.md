
# :rocket: [UNav](https://github.com/endeleze/UNav)

[English](README.md) **|** [简体中文](README_CN.md)**|** [แบบไทย](README_Thai.md)

---

UNav is a vision-based location system designed to assist visually impaired individuals in navigating unfamiliar environments.

## :sparkles: New Features

- May 29, 2023. Support **Parallel RanSAC** computing 

<details>
  <summary>More</summary>

</details>

## :wrench: Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.13](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

1. Clone repo

    ```bash
    git clone https://github.com/endeleze/UNav.git
    ```

1. Install dependent packages

    ```bash
    cd UNav
    pip install -r requirements.txt
    ```
## :computer: Using
1. Server-Client

    * Setup [server.yaml](configs/server.yaml) and tune [hloc.yaml](configs/hloc.yaml) us needed.

   * Put the data into **IO_root** you defined as following structure
   
      ```bash
      UNav-IO/
      ├── data
      │   ├── destination.json
      │   ├── PLACE
      │   │   └── BUILDING
      │   │       └── FLOOR
      │   │           ├── access_graph.npy
      │   │           ├── boundaries.json
      │   │           ├── feats-superpoint.h5
      │   │           ├── global_features.h5
      │   │           ├── topo-map.json
      │   │           └── floorplan.png
      ```

      Note that you need to rerun [Path_finder_waypoints.py](./Path_finder_waypoints.py) using **step2_automatically.sh** if you do not have ***access_graph.npy***
    * Run server using
      ```bash
      source shell/server.sh
      ```
    * Run client device
      * Jetson Board
      * Android
  
2. Visualization-GUI
    TODO

Note that UNav is only tested in Ubuntu, and may be not suitable for Windows or MacOS.

## :hourglass_flowing_sand: TODO List

Please see [project boards](https://github.com/endeleze/UNav/projects).



## :e-mail: Contact

If you have any question, please email `ay1620@nyu.edu`.
