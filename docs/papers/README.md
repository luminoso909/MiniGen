## **一、S1 两周核心论文**

固定版本核验与下载日期：2026-10-24。原文为第三方论文，保留作者与 arXiv 来源，不标作项目原创。文件大小、SHA-256、页数、作者和固定版本见 [manifest.json](manifest.json)；三个 PDF 已逐页解析，身份按首页标题核对。

| 阅读层次 | 本地论文 | 原始来源与版本 | 读哪里 |
| --- | --- | --- | --- |
| 第一周主读，第二周复习 | [DDPM](DDPM_Denoising_Diffusion_Probabilistic_Models_2020_v2.pdf)，25 页 | [Ho 等，2020，2006.11239v2](https://arxiv.org/abs/2006.11239v2) | 第 2–3 节，算法 1/2；附录推导按需回看 |
| 第二周主读 | [Auto-Encoding Variational Bayes](VAE_Auto_Encoding_Variational_Bayes_1312.6114v11.pdf)，14 页 | [Kingma、Welling，2013，1312.6114v11](https://arxiv.org/abs/1312.6114v11) | 第 2–3 节、附录 B/C；v11 为 2022 年修订版本，不能把修订年份当首次提出年份 |
| 第二周桥梁选读 | [Latent Diffusion Models](LDM_High_Resolution_Image_Synthesis_2112.10752v2.pdf)，45 页 | [Rombach 等，CVPR 2022，2112.10752v2](https://arxiv.org/abs/2112.10752v2) | 图 1、第 3.1–3.2 节、第 4.1 节；本周不要求读完附录或复现 |

## **二、边读边做：三张阅读卡**

### **1. DDPM：训练目标怎样对应反向生成**

问题：怎样学习多步生成过程。关键连接：固定前向链、可计算后验、噪声参数化和变分训练。你需要把公式对应到加噪、预测噪声和反向一步，解释为什么训练可以随机抽一个时间而不展开整条链。

证据入口：[第一周 Notebook](../learning/S1_第1周_DDPM原理与MNIST实验.ipynb) 与 [第一周实验](../../reports/stage1/week1.md)。局限：论文的大规模结果与本项目 MNIST 短训不同；DDPM 简化 MSE 不能与 VAE 重建 BCE 比原始数值。

### **2. VAE：近似后验与可微采样为什么一起出现**

问题：潜变量积分与后验难算。关键改动：使用可训练近似后验，把采样改写为独立噪声上的确定变换。你需要独立推导 ELBO 与高斯 KL，分清损失里的先验 KL 和下界差距里的真实后验 KL。

证据入口：[第二周 Notebook](../learning/S1_第2周_VAE原理与潜空间实验.ipynb) 与 [个人核心任务](../../learning/L02_VAE核心任务.md)。局限：灰度 BCE 是本项目的代理目标，低重建误差不证明先验采样质量。

### **3. LDM：压缩为什么既省计算，也可能丢能力**

问题：像素扩散计算成本高。关键设计：先学可重建的压缩表示，再在潜空间做扩散。你需要画出编码、潜变量去噪和解码路径，说明压缩率、细节保留与计算成本怎样相互约束。

证据入口：本周 Notebook 第十一章。下一阶段才运行冻结 Autoencoder 的属性保留验证；本周没有使用其权重或复现论文。MNIST MLP VAE 与现代空间 Autoencoder 的目标、结构和表示维度不同。
