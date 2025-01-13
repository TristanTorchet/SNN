
// Create at table

Cramer 2020 - vFF - 1or2or3 FF - 48.3
Cramer 2020 - vRSNN - 3 R - 71.4
Cramer 2020 - vRSNN+BE - 3 R (best effort) - 83.2
Perez 2020 - HetRSNN - 700-128-20 R with heterogeneous tau_mem - 82
B. Yin 2020 - Effective and efficient computation with multiple-timescale spiking recurrent neural networks - 84.4 - 141.3k - ICONS 2020 - sMNIST - 156.3k - 98.7 - psMNIST - 156.3k - 94.3
B. Yin 2021 - ALIF - Accurate and efficient time domain classification with adaptive SRNN - 90.4 - ... - Nature Machine Intelligence 
M. Yao 2021 - TA-SNN - Temporal-wise attention SNN for event streams classification - 91.08 - 121.7k - IEEE ICCV
Zenke 2021 - LIF - The remarkable robustness of SG learning for instilling complex function in SNN - 84 - 249k - Neural Computations
Bittar 2022 - RadLIF - A SG spiking baseline - 94.62 - 3.9M - Frontiers in NSc 
P. Sun 2023 - Adaptive axonal delay - Adapative axonal delays in FF SNN for accurate spoken word recognition - 92.45 - 109.1K - IEEE ICASSP 2023
Z. Wang Feb 2023 - ASGL -  Adaptive smoothing gradient learning for SNN - 87.9 - 230.4K - ICLR 2023 (Rejected)
S. Zhang Aug 2023 - Tc-lif: a atwo compartment spiking neuron model for long-term sequential modelling" - 88.91 - 141.8k - AAAI 2024 - sMNIST - 151.4k - 99.2 - psMNIST - 151.4k - 95.36
W. Fang Sep 2023 - PSN - Parallel spiking neuron with high efficiency and ability to learn long dependencies - 89.75 - 232.5k - NeurIPS 2024 - sMNIST - 2.5M - 97.9 - psMNIST - 2.5M - 97.76
X. Chen Sep 2023 - PMSN - A Paralle Multi-compartment Spiking Neuron for Multi-scale temporal Processing -  94.25/95.1 - 120.3k/199.3k - ICLR 2024 Rejected - sMNIST - 66.3k/156.4k - 99.4/99.53 - psMNIST - 66.3k/156.4k - 97.16/97.78

Boijan Yin 2020 - https://scholar.google.com/citations?hl=en&user=5fpuCPYAAAAJ&view_op=list_works&sortby=pubdate 
- AdLIF with delta synapses
- dt=4ms or 10ms
- SNN with ReLU
- 2 layers of 128 
- SG: gaussian N(vth, std=0.5)
- Increasing layer width increases sparsity
- github
- Speak about energy 
- sMNIST, SHD, 
- loss max over time
- No validation set 

B. Yin 2021
- sMNIST, SHD, psMNIST, soLI, TIMIT
- multiple SG -> best MultiGaussian (h=0.15, s=6, std=0.5) reaches 90.4
- loss max over time
- Weight initialisation: ff Xavier, rec Orthogonal
- Tau initialisation: normal tau_mem (20,5), tau_adp(150,10)
- Initialize Vmem(t=0) randomly uniformly([0,vth])
- Selected best Test Accuracy ...
- vrest = 0
- (only 20 epochs not sure)
- bs=64
- github




LIF 28 - TCLIF
PLIF 28 
GLIF 28
DEXAT 26 - A. Shaban An adaptive threshold neuron for RSNN with nanodevice HW implementation - Nat Communication 2021


Mark shoene 

Yulong Huang - CLIF: Complementary Leaky Integrate-and-Fire Neuron for Spiking Neural Networks - ICLR 2024
Yulong Huang - PRF: PARALLEL RESONATE AND FIRE NEURON FOR LONG SEQUENCE LEARNING IN SPIKING NEURAL NET- WORKS - ICLR 2025 (submitted)
TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting - ICLR 2025 (submitted)



Don't forget Github and safari 
FROM Notes in MACBOOK
SNN SSM: 
- Spiking Structured State Space Model for Monaural Speech Enhancement, Sep 23
Yu Du, Xu Liu, Yansong Chua
- Rethinking Spiking Neural Networks as State Space Models, Jun 24
Malyaban Bal, Abhronil Sengupta
- PSN, Apr 23 (not SSM but parallel)
- SpikingSSM, Aug 24
- PMSN, 27 Aug 24


From PMSN: 

PMSN: A Parallel Multi-compartment Spiking Neuron for Multi-scale Temporal Processing
- Xinyi Chen, Jibin Wu, Chenxiang Ma, Yinsong Yan, Yujie Wu, 
- Kay Chen Tan PI 17k, no NIPS, ICLR, ICML

GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks
- Xingting Yao, 
- Fanrong Li, 
- Zitao Mo, 
- Jian Cheng

Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies
- Wei Fang, machine ICLR…,  scholar https://scholar.google.com/citations?hl=fr&user=e2lED2gAAAAJ&view_op=list_works&sortby=pubdate
- Zhaofei Yu, Assistant Professor, time constant training…, exPhD of Tian, visited Maass for a year
- Zhaokun Zhou, secondary PhD
- Ding Chen, Shanghai Jiaotong University, MS student
- Yanqi Chen, PhD, most relevant (dendrites, pruning)
- Zhengyu Ma, no info
- Timothée Masquelier, 
- Yonghong Tian, PI 13k 
- Pekin University

Deep residual learning in spiking neural networks 
- Wei Fang, 
- Zhaofei Yu, 
- Yanqi Chen, 
- Tiejun Huang, Full prof, not ml
- Timothée Masquelier, 
- Yonghong Tian

Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting
- Shikuang Deng, 
- Yuhang Li, 
- Shanghang Zhang, PI, OOD Generalisation
- Shi Gu
(Not from PMSN) Surrogate module learning: Reduce the gradient error accumulation in training spiking neural networks
- Shikuang Deng, PhD student
- Hao Lin, Master Student
- Yuhang Li, Research Assistant in China, now PhD at Yale
- Shi Gu, PI 4k
- University of Electronic Science and Technology of China

Membrane potential batch normalization for spiking neural networks
- Yufei Guo, PostDoc, machine NIPS, …
- Yuhan Zhang, PostDoc, only one publi in open review
- Yuanpei Chen, no info
- Weihang Peng, no info

InfLoR-SNN: Reducing Information Loss for Spiking Neural Networks
- Y Guo, Y Chen, L Zhang, X Liu, X Tong, Y Ou, X Huang, Z Ma





From SpikingSSM

SpikingSSMs: Learning Long Sequences with Sparse and Parallel Spiking State Space Models
- Shuaijie Shen, PhD of Yonghong Tian, Huaweri
- Chao Wang, 
- Renzhuo Huang, 
- Yan Zhong, 
- Qinghai Guo, 
- Zhichao Lu, 
- Jianguo Zhang, 
- Luziwei Leng
- Chinese academy of science

Efficient Deep Spiking Multilayer Perceptrons With Multiplication-Free Inference
- Boyan Li, Luziwei Leng, Shuaijie Shen (SpikingSSM), Kaixuan Zhang, Jianguo Zhang, Jianxing Liao, Ran Cheng
- 9
Efficient training spiking neural networks with parallel spiking unit
- Yang Li, Yinqian Sun, Xiang He, Yiting Dong, Dongcheng Zhao, Yi Zeng
- Other interesting titles from Yi Zeng (PI) 
    - Directly training temporal Spiking Neural Network with sparse surrogate gradient
    - Matrix-transformation based low-rank adaptation (mtlora): A brain-inspired method for parameter-efficient fine-tuning
