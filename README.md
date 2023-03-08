# Multimodal Prompting with Missing Modalities for Visual Recognition (CVPR 2023)
Official PyTorch implementaton of CVPR 2023 paper "Multimodal Prompting with Missing Modalities for Visual Recognition".  
You can visit our project website [here](https://yilunlee.github.io/missing_aware_prompts/).

## Introduction
In this paper, we tackle two challenges in multimodal learning for visual recognition: 1) when missing-modality occurs either during training or testing in real-world situations; and 2) when the computation resources are not available to finetune on heavy transformer models. To this end, we propose to utilize prompt learning and mitigate the above two challenges together. Specifically, our modality-missing-aware prompts can be plugged into multimodal transformers to handle general missing-modality cases, while only requiring less than 1% learnable parameters compared to training the entire model. 

<div align="center">
  <img src="fig/model.jpeg"/>
</div>

## Usage
To be released soon...
### Enviroment

### Prepare Dataset

### Evaluation

### Train

## Citation
If you find this work useful for your research, please cite:

```Bibtex
@inproceedings{lee2023cvpr,
 title = {Multimodal Prompting with Missing Modalities for Visual Recognition},
 author = {Yi-Lun Lee and Yi-Hsuan Tsai and Wei-Chen Chiu and Chen-Yu Lee},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 year = {2023}
}
```

## Acknowledgements
This code is based on [ViLT](https://github.com/dandelin/ViLT.git).