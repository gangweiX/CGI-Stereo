<p align="center">
  <h1 align="center">CGI-Stereo: Accurate and Real-Time Stereo Matching via Context and
Geometry Interaction</h1>
  <p align="center">
    Gangwei Xu, Huan Zhou, Xin Yang
  </p>
  <h3 align="center"><a href="https://arxiv.org/pdf/2301.02789.pdf">Paper</a>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://github.com/gangweiX/CGI-Stereo/blob/main/imgs/CGI-network.jpg" alt="Logo" width="100%">
  </a>
</p>


# How to use

## Environment
* Python 3.8
* Pytorch 1.12

## Install

### Create a virtual environment and activate it.

```
conda create -n CGI python=3.8
conda activate CGI
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install timm==0.5.4
```
