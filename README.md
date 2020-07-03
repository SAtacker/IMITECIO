# IMITECIO

![Imitecio](https://github.com/SAtacker/IMITECIO/blob/master/docs/images/ImitecioLogoFree.png)  

Pose Estimation (& Replication) Based Game  

<!-- TABLE OF CONTENTS -->

## Table of Contents

* [About the Project](#about-the-project)
  * [Tech Stack](#tech-stack)
  * [File Structure](#file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Results and Demo](#results-and-demo)
* [Future Work](#future-work)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [Acknowledgements and Resources](#acknowledgements-and-resources)
* [License](#license)

<!-- ABOUT THE PROJECT -->

## About The Project

*  Pose Estimation has been for a while in the deep learning industry. Considering the various applications of it, we aimed at creating a simple game at first. Some few things we intend to do further are mentioned at the end. 
Refer this [documentation](https://link/to/report/)

### Tech Stack

* The [Posenet model](https://github.com/tensorflow/tfjs-models/tree/master/posenet) trained by Google is used.
* For the Posenet Js to Python port [this](https://github.com/rwightman/posenet-python)  repo is used.
<!-- This section should list the technologies you used for this project. Leave any add-ons/plugins for the prerequisite section. Here are a few examples. -->

The rest is openCV for webcam input, Unity for NotEndlessRunner game and keyboard for keypresses. Although pynput is used , we prefer pywin32 for windows.

* [OpenCV](https://opencv.org/)
* [Unity](https://unity.com/)  

### File Structure

```
IMITECIO
├───docs
│   └───images
├───NotEndlessRunner
│   |
│   ├───Assets
│   │   ├───Animation
│   │   ├───Materials
│   │   ├───Prefab
│   │   ├───Raw Mocap Data
│   │   ├───Scenes
│   │   ├───Scripts
│   │   ├───Sprites
│   │   └───TextMesh Pro
│   │       ├───Documentation
│   │       ├───Fonts
│   │       ├───Resources
│   │       └───Sprites
│   ├───Library
|   |
│   ├───Logs
│   ├───obj
│   ├───Packages
│   └───ProjectSettings
├───OsX.app
│   └───Contents
│       ├───Frameworks
│       ├───MacOS
│       ├───MonoBleedingEdge
│       └───Resources
└───posenet
    ├───MonoBleedingEdge
    │   ├───EmbedRuntime
    │  
    ├───NotEndlessRunner_Data
    │   ├───Managed
    │   └───Resources
    ├───posenet
    │   └───converter
    └───models                                         ### You may need to download the models initially.
        └───checkpoints
```

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites & Installation

* Tested on Windows

```sh
git clone https://github.com/SAtacker/IMITECIO.git
cd IMITECIO
```

In your Virtual Environment

```cmd
pip install -r "requirements.txt"
python posenet\webcam_demo.py
```

<!-- RESULTS AND DEMO -->
## Results and Demo

![**NotEndlessRunner**](https://github.com/SAtacker/IMITECIO/blob/master/docs/images/GameScreenshot.png)

* On Manipulating the code according to your requirements, you could use it to control most of the games.

![**Part 1**](https://github.com/SAtacker/IMITECIO/blob/master/docs/images/GameControlsPart1.gif)  
![**Part 2**](https://github.com/SAtacker/IMITECIO/blob/master/docs/images/GameControlsPart2.gif)

* We have tried playing with [SpaceInvaders](https://github.com/leerob/Space_Invaders)

![**Space Invaders**](https://github.com/SAtacker/IMITECIO/blob/master/docs/images/SpaceInvaders.PNG)  

<!-- FUTURE WORK -->
## Future Work

- [x] Posenet tflite to Python - Single Person detection
- [x] Posenet Based Game
- [ ] Pose Estimation in 3d
- [ ] Socket Code instead of Keyboard Presses
- [ ] C++ Port for faster recognition


<!-- TROUBLESHOOTING -->
## Troubleshooting

* You may need to install tensorflow 1.15.0 or tensorflow 1.13 on your own.
* There could be an issue only in the tensorflow installation.
* However one small thing to remember is that multithreading does not guarentee the Game to run first . Meaning that the OpenCV window might appear later and gamethread before. We dont't have a fix for this. Rather press Alt + Tab to overcome this thing.

<!-- CONTRIBUTORS -->
## Contributors

* [Shreyas Atre](https://github.com/SAtacker)
* [Dhruvi Doshi](https://github.com/dhruvi29)
* [Sarah Nisar](https://github.com/sarah-nisar)
* [Arnav](https://github.com/wh1t3-h4t)

<!-- ACKNOWLEDGEMENTS AND REFERENCES -->
## Acknowledgements and Resources

* [SRA VJTI](http://sra.vjti.info/) Eklavya 2020  
* Refered [this](https://github.com/rwightman/posenet-python) for PoseNet Python
Some More
* [PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet)
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* [Pytorch 3D pose](https://github.com/xingyizhou/Pytorch-pose-hg-3d)
* [Paper](https://arxiv.org/pdf/1312.4659)
* [Paper](https://arxiv.org/pdf/1603.06937.pdf)
* [Paper](https://arxiv.org/pdf/1701.01779.pdf)
* [Paper](https://arxiv.org/pdf/1803.08225.pdf)
* [Space Invaders](https://github.com/leerob/Space_Invaders)

<!-- LICENSE -->
## License

[License](LICENSE)
