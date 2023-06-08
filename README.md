# IEEE TNSRE_23
[Automated Sleep Staging via Parallel Frequency-Cut Attention](https://ieeexplore.ieee.org/abstract/document/10041186)

Zheng Chen, Ziwei Yang, Lingwei Zhu, Wei Chen, Toshiyo Tamura, Naoaki Ono, MD Altaf-Ul-Amin, Shigehiko Kanaya and Ming Huang

In this paper, we propose a novel framework that is based on authoritative guidance in sleep medicine and is designed to automatically capture the time-frequency characteristics of sleep electroencephalogram (EEG) signals in order to make staging decisions. Our framework consists of two main phases: a feature extraction process that partitions the input EEG spectrograms into a sequence of time-frequency patches, and a staging phase that searches for correlations between the extracted features and the defining characteristics of sleep stages. To model the staging phase, we utilize a Transformer model with an attention-based module, which allows for the extraction of global contextual relevance among time-frequency patches and the use of this relevance for staging decisions. 

---------------------------------------------------------------------------------------------------------------------


System overview             |  Time-frequency patching
:-------------------------:|:-------------------------:
![alt text](https://github.com/chenzRG/TNSRE_23/assets/125750017/cf865ab2-f0ae-4854-942c-3ff95d3db0c0)  | <img width="1200" alt="image" src="https://github.com/chenzRG/TNSRE_23/assets/125750017/8b2ab0c7-f696-4b06-a30f-e04774f11153">

Fequencey-band-time-index visualization resutls [1],[2].

<p align="center">
<img width="900" alt="image" src="https://user-images.githubusercontent.com/34312998/133877630-9b2f2eec-11e0-4d41-8c36-5afd02dd78d6.png">
</p>




## Setup

You can install the required dependencies using pip.

```bash
pip install -r requirements.txt
```

If you're using other than CUDA 10.2, you may need to install PyTorch for the proper version of CUDA. See [instructions](https://pytorch.org/get-started/locally/) for more details.

## Training

```python
python main.py 
--epochs=xx
...
--dim=xx
--layers=xx
...
--dic_name <path/to/model_checkpoint> 
<model_checkpoint.pth>
```


## Reference

[1] H. Chefer, S. Gur, and L. Wolf, “Generic attention-model explainability for interpreting bi-modal and encoder-decoder transformers,” CoRR, vol. abs/2103.15679, 2021. [Online]. Available: https://arxiv.org/abs/2103.15679

[2] Hila Chefer, Shir Gur, Lior Wolf, “Transformer interpretability beyond attention visualization,” arXiv preprint arXiv:2012.09838, 2020.

## Citation
If you find this code useful in your research, please consider citing:

    @ARTICLE{TNSREchen23,
  	author={Chen, Zheng and Yang, Ziwei and Zhu, Lingwei and Chen, Wei and Tamura, Toshiyo and Ono, Naoaki and Altaf-Ul-Amin, Md and Kanaya, Shigehiko and Huang, Ming},
  	journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
  	title={Automated Sleep Staging via Parallel Frequency-Cut Attention}, 
  	year={2023},
  	volume={31},
  	pages={1974-1985},
  }




