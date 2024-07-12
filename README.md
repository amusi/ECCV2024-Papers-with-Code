# ECCV 2024 论文和开源项目合集(Papers with Code)

ECCV 2024 decisions are now available！


> 注1：欢迎各位大佬提交issue，分享ECCV 2024论文和开源项目！
>
> 注2：关于往年CV顶会论文以及其他优质CV论文和大盘点，详见： https://github.com/amusi/daily-paper-computer-vision
>
> - [CVPR 2024](https://github.com/amusi/CVPR2024-Papers-with-Code)
> - [ECCV 2022](ECCV2022-Papers-with-Code.md)
> - [ECCV 2020](ECCV2020-Papers-with-Code.md)
> - 

欢迎扫码加入【CVer学术交流群】，这是最大的计算机视觉AI知识星球！每日更新，第一时间分享最新最前沿的计算机视觉、AI绘画、图像处理、深度学习、自动驾驶、医疗影像和AIGC等方向的学习资料，学起来！

![](CVer学术交流群.png)

# 【ECCV 2024 论文开源目录】

- [3DGS(Gaussian Splatting)](#3DGS)
- [Mamba / SSM)](#Mamba)
- [Avatars](#Avatars)
- [Backbone](#Backbone)
- [CLIP](#CLIP)
- [MAE](#MAE)
- [Embodied AI](#Embodied-AI)
- [GAN](#GAN)
- [GNN](#GNN)
- [多模态大语言模型(MLLM)](#MLLM)
- [大语言模型(LLM)](#LLM)
- [NAS](#NAS)
- [OCR](#OCR)
- [NeRF](#NeRF)
- [DETR](#DETR)
- [Prompt](#Prompt)
- [扩散模型(Diffusion Models)](#Diffusion)
- [ReID(重识别)](#ReID)
- [长尾分布(Long-Tail)](#Long-Tail)
- [Vision Transformer](#Vision-Transformer)
- [视觉和语言(Vision-Language)](#VL)
- [自监督学习(Self-supervised Learning)](#SSL)
- [数据增强(Data Augmentation)](#DA)
- [目标检测(Object Detection)](#Object-Detection)
- [异常检测(Anomaly Detection)](#Anomaly-Detection)
- [目标跟踪(Visual Tracking)](#VT)
- [语义分割(Semantic Segmentation)](#Semantic-Segmentation)
- [实例分割(Instance Segmentation)](#Instance-Segmentation)
- [全景分割(Panoptic Segmentation)](#Panoptic-Segmentation)
- [医学图像(Medical Image)](#MI)
- [医学图像分割(Medical Image Segmentation)](#MIS)
- [视频目标分割(Video Object Segmentation)](#VOS)
- [视频实例分割(Video Instance Segmentation)](#VIS)
- [参考图像分割(Referring Image Segmentation)](#RIS)
- [图像抠图(Image Matting)](#Matting)
- [图像编辑(Image Editing)](#Image-Editing)
- [Low-level Vision](#LLV)
- [超分辨率(Super-Resolution)](#SR)
- [去噪(Denoising)](#Denoising)
- [去模糊(Deblur)](#Deblur)
- [自动驾驶(Autonomous Driving)](#Autonomous-Driving)
- [3D点云(3D Point Cloud)](#3D-Point-Cloud)
- [3D目标检测(3D Object Detection)](#3DOD)
- [3D语义分割(3D Semantic Segmentation)](#3DSS)
- [3D目标跟踪(3D Object Tracking)](#3D-Object-Tracking)
- [3D语义场景补全(3D Semantic Scene Completion)](#3DSSC)
- [3D配准(3D Registration)](#3D-Registration)
- [3D人体姿态估计(3D Human Pose Estimation)](#3D-Human-Pose-Estimation)
- [3D人体Mesh估计(3D Human Mesh Estimation)](#3D-Human-Pose-Estimation)
- [医学图像(Medical Image)](#Medical-Image)
- [图像生成(Image Generation)](#Image-Generation)
- [视频生成(Video Generation)](#Video-Generation)
- [3D生成(3D Generation)](#3D-Generation)
- [视频理解(Video Understanding)](#Video-Understanding)
- [行为检测(Action Detection)](#Action-Detection)
- [文本检测(Text Detection)](#Text-Detection)
- [知识蒸馏(Knowledge Distillation)](#KD)
- [模型剪枝(Model Pruning)](#Pruning)
- [图像压缩(Image Compression)](#IC)
- [三维重建(3D Reconstruction)](#3D-Reconstruction)
- [深度估计(Depth Estimation)](#Depth-Estimation)
- [轨迹预测(Trajectory Prediction)](#TP)
- [车道线检测(Lane Detection)](#Lane-Detection)
- [图像描述(Image Captioning)](#Image-Captioning)
- [视觉问答(Visual Question Answering)](#VQA)
- [手语识别(Sign Language Recognition)](#SLR)
- [视频预测(Video Prediction)](#Video-Prediction)
- [新视点合成(Novel View Synthesis)](#NVS)
- [Zero-Shot Learning(零样本学习)](#ZSL)
- [立体匹配(Stereo Matching)](#Stereo-Matching)
- [特征匹配(Feature Matching)](#Feature-Matching)
- [场景图生成(Scene Graph Generation)](#SGG)
- [隐式神经表示(Implicit Neural Representations)](#INR)
- [图像质量评价(Image Quality Assessment)](#IQA)
- [视频质量评价(Video Quality Assessment)](#Video-Quality-Assessment)
- [数据集(Datasets)](#Datasets)
- [新任务(New Tasks)](#New-Tasks)
- [其他(Others)](#Others)

<a name="3DGS"></a>

# 3DGS(Gaussian Splatting)

**MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images**

- Project: https://donydchen.github.io/mvsplat
- Paper: https://arxiv.org/abs/2403.14627
- Code：https://github.com/donydchen/mvsplat

**CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians**

- Paper: https://arxiv.org/abs/2404.01133
- Code: https://github.com/DekuLiuTesla/CityGaussian

**FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting**

- Project: https://zehaozhu.github.io/FSGS/
- Paper: https://arxiv.org/abs/2312.00451
- Code: https://github.com/VITA-Group/FSGS



<a name="Mamba"></a>

# Mamba / SSM

**VideoMamba: State Space Model for Efficient Video Understanding**

- Paper: https://arxiv.org/abs/2403.06977

- Code: https://github.com/OpenGVLab/VideoMamba

<a name="Avatars"></a>

# Avatars





<a name="Backbone"></a>

# Backbone



<a name="CLIP"></a>

# CLIP





<a name="MAE"></a>

# MAE

<a name="Embodied-AI"></a>

# Embodied AI



<a name="GAN"></a>

# GAN

<a name="OCR"></a>

# OCR

**Bridging Synthetic and Real Worlds for Pre-training Scene Text Detectors**

- Paper: https://arxiv.org/pdf/2312.05286

- Code: https://github.com/SJTU-DeepVisionLab/FreeReal 



<a name="Occupancy"></a>

# Occupancy

**Fully Sparse 3D Occupancy Prediction**

- Paper: https://arxiv.org/abs/2312.17118
- Code: https://github.com/MCG-NJU/SparseOcc



<a name="NeRF"></a>

# NeRF





<a name="DETR"></a>

# DETR



<a name="Prompt"></a>

# Prompt

<a name="MLLM"></a>

# 多模态大语言模型(MLLM)

**SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant**

- Paper: https://arxiv.org/abs/2403.11299
- Code: https://github.com/heliossun/SQ-LLaVA

**ControlCap: Controllable Region-level Captioning**

- Paper: https://arxiv.org/abs/2401.17910
- Code: https://github.com/callsys/ControlCap 

<a name="LLM"></a>

# 大语言模型(LLM)



<a name="NAS"></a>

# NAS

<a name="ReID"></a>

# ReID(重识别)



<a name="Diffusion"></a>

# 扩散模型(Diffusion Models)





<a name="Vision-Transformer"></a>

# Vision Transformer

**GiT: Towards Generalist Vision Transformer through Universal Language Interface**

- Paper: https://arxiv.org/abs/2403.09394
- Code: https://github.com/Haiyang-W/GiT

<a name="VL"></a>

# 视觉和语言(Vision-Language)

**GalLoP: Learning Global and Local Prompts for Vision-Language Models**

- Paper：https://arxiv.org/abs/2407.01400

<a name="Object-Detection"></a>

# 目标检测(Object Detection)



<a name="Anomaly-Detection"></a>

# 异常检测(Anomaly Detection)



<a name="VT"></a>

# 目标跟踪(Object Tracking)





<a name="Semantic-Segmentation"></a>

# 语义分割(Semantic Segmentation)

**Context-Guided Spatial Feature Reconstruction for Efficient Semantic Segmentation**

- Paper: https://arxiv.org/abs/2405.06228

- Code: https://github.com/nizhenliang/CGRSeg

<a name="MI"></a>

# 医学图像(Medical Image)



<a name="MIS"></a>

# 医学图像分割(Medical Image Segmentation)



<a name="VOS"></a>

# 视频目标分割(Video Object Segmentation)

**DVIS-DAQ: Improving Video Segmentation via Dynamic Anchor Queries**

- Project: https://zhang-tao-whu.github.io/projects/DVIS_DAQ/
- Paper: https://arxiv.org/abs/2404.00086
- Code: https://github.com/zhang-tao-whu/DVIS_Plus 

<a name="Autonomous-Driving"></a>

# 自动驾驶(Autonomous Driving)

**Fully Sparse 3D Occupancy Prediction**

- Paper: https://arxiv.org/abs/2312.17118
- Code: https://github.com/MCG-NJU/SparseOcc

**milliFlow: Scene Flow Estimation on mmWave Radar Point Cloud for Human Motion Sensing**

- Paper: https://arxiv.org/abs/2306.17010
- Code link: https://github.com/Toytiny/milliFlow/ 

<a name="3D-Point-Cloud"></a>

# 3D点云(3D-Point-Cloud)



<a name="3DOD"></a>

# 3D目标检测(3D Object Detection)

**3D Small Object Detection with Dynamic Spatial Pruning**

- Project: https://xuxw98.github.io/DSPDet3D/
- Paper: https://arxiv.org/abs/2305.03716
- Code: https://github.com/xuxw98/DSPDet3D 

<a name="3DOD"></a>

# 3D语义分割(3D Semantic Segmentation)

<a name="Image-Editing"></a>

# 图像编辑(Image Editing)





<a name="Image-Inpainting"></a>

# 图像补全/图像修复(Image Inpainting)

**BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion**

- Project https://tencentarc.github.io/BrushNet/
- Paper: https://arxiv.org/abs/2403.06976
- Code: https://github.com/TencentARC/BrushNet

<a name="Video-Editing"></a>

# 视频编辑(Video Editing)



<a name="LLV"></a>

# Low-level Vision



# 超分辨率(Super-Resolution)



<a name="Denoising"></a>

# 去噪(Denoising)

## 图像去噪(Image Denoising)

<a name="3D-Human-Pose-Estimation"></a>

# 3D人体姿态估计(3D Human Pose Estimation)



<a name="Image-Generation"></a>

# 图像生成(Image Generation)

**Object-Conditioned Energy-Based Attention Map Alignment in Text-to-Image Diffusion Models**

- Paper: https://arxiv.org/abs/2404.07389
- Code: https://github.com/YasminZhang/EBAMA 

<a name="Video-Generation"></a>

# 视频生成(Video Generation)





<a name="3D-Generation"></a>

# 3D生成



<a name="Video-Understanding"></a>

# 视频理解(Video Understanding)

**VideoMamba: State Space Model for Efficient Video Understanding**

- Paper: https://arxiv.org/abs/2403.06977

- Code: https://github.com/OpenGVLab/VideoMamba

<a name="KD"></a>

# 知识蒸馏(Knowledge Distillation)



<a name="Stereo-Matching"></a>

# 立体匹配(Stereo Matching)



<a name="SGG"></a>

# 场景图生成(Scene Graph Generation)



<a name="Video-Quality-Assessment"></a>

# 视频质量评价(Video Quality Assessment)

<a name="Datasets"></a>

# 数据集(Datasets)



# 其他(Others)


