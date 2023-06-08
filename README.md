# IEEE TNSRE_23
[Automated Sleep Staging via Parallel Frequency-Cut Attention](https://ieeexplore.ieee.org/abstract/document/10041186)

Zheng Chen, Ziwei Yang, Lingwei Zhu, Wei Chen, Toshiyo Tamura, Naoaki Ono, MD Altaf-Ul-Amin, Shigehiko Kanaya and Ming Huang


This work proposed a novel framework designed on top of authoritative sleep medicine guidance that can automatically capture the time-frequency nature of sleep electroencephalogram (EEG) signals and make staging decisions.

System overview             |  Time-frequency patching
:-------------------------:|:-------------------------:
![alt text](https://github.com/chenzRG/TNSRE_23/assets/125750017/cf865ab2-f0ae-4854-942c-3ff95d3db0c0)  | <img width="1200" alt="image" src="https://github.com/chenzRG/TNSRE_23/assets/125750017/8b2ab0c7-f696-4b06-a30f-e04774f11153">

Fequencey-band-time-index visualization resutls [2],[3].

![alt text](https://user-images.githubusercontent.com/34312998/133877630-9b2f2eec-11e0-4d41-8c36-5afd02dd78d6.png)



## Setup

You can install the required dependencies using pip.

```bash
pip install -r requirements.txt
```

If you're using other than CUDA 10.2, you may need to install PyTorch for the proper version of CUDA. See [instructions](https://pytorch.org/get-started/locally/) for more details.

## Description

The proposed methods can be found in _Model_EEGspec-VTrans.ipynb.ipynb_ and _Model_EEGspec-Seq-VTrans.ipynb_.

Gradient-attention-based visualization has shown in _Vit_visualization.ipynb_.

_Model_#_xxxxx_ presents the ablations in different architecture.

## Reference

[1] Z. Chen, K. Odani, P. Gao, N. Ono, M. Altaf-Ul-Amin, S. Kanaya, and M. Huang, “Feasibility analysis of transformer model for eeg-based sleep scoring,” in 2021 IEEE International Conference on Biomedical and Health Informatics, Virtual, July 2021.

[2] H. Chefer, S. Gur, and L. Wolf, “Generic attention-model explainability for interpreting bi-modal and encoder-decoder transformers,” CoRR, vol. abs/2103.15679, 2021. [Online]. Available: https://arxiv.org/abs/2103.15679

[3] Hila Chefer, Shir Gur, Lior Wolf, “Transformer interpretability beyond attention visualization,” arXiv preprint arXiv:2012.09838, 2020.
