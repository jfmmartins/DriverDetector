# Driver Monitoring System (DMS) – Eye State Detection

## Overview

This project is a **Driver Monitoring System (DMS)** that detects whether a driver's eyes are open or closed in real-time.
The system aims to improve **road safety** by identifying potential driver drowsiness, a major cause of accidents worldwide.
In this project the job of a DMS camera is simulated yo test if the results can be achieved by using a non dedicated camera for it, like a smartphone camera.

✅ Real-time detection

✅ Eye state classification (Open / Closed)

✅ Based on Computer Vision and Deep Learning

## Demo

TBR

## Tech Stack

**Programming Language**: Python

**Libraries**: OpenCV, TensorFlow/PyTorch, NumPy

**Environment**: Python Scripts

**Hardware**: Works with any standard webcam

## Project Structure

```
📂 DriverDetector
    ├── app/             
    │    ├── models/     
    │    ├── webcam.py  
    │    ├── ml.py  
    │    └── main.py
    └── README.md
```

## Installation & Usage

1 - Clone The repository

```bash
git clone https://github.com/jfmmartins/DriverDetector.git
cd DriverDetector
```

2 - Install dependecies

```bash
pip install uv
```

3 - Run the application

```bash
uv run main.py
```


## Results

TBD

## Future Improvements

- Yawn detection

- Head pose estimation

- Integration with car alert systems