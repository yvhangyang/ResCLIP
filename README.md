<div align="center">
<h1>ResCLIP </h1>
<h3>ResCLIP: Residual Attention for Training-free Dense Vision-language Inference</h3>
<div>
    <h4 align="center">
        <a href='https://arxiv.org/abs/2411.15851'><img src='https://img.shields.io/badge/ArXiv-2411.15851-red'></a>
    </h4>
</div>
</div>

## News
* **` Nov. 23rd, 2024`**: We release paper for ResCLIP.
* Code will be released.

## Abstract
While vision-language models like CLIP have shown remarkable success in open-vocabulary tasks, their application is currently confined to image-level tasks, and they still struggle with dense predictions. Recent works often attribute such deficiency in dense predictions to the self-attention layers in the final block, and have achieved commendable results by modifying the original query-key attention to self-correlation attention, (e.g., query-query and key-key attention). However, these methods overlook the cross-correlation attention (query-key) properties, which capture the rich spatial correspondence. In this paper, we reveal that the cross-correlation of the self-attention in CLIP's non-final layers also exhibits localization properties. Therefore, we propose the Residual Cross-correlation Self-attention (RCS) module, which leverages the cross-correlation self-attention from intermediate layers to remold the attention in the final block. The RCS module effectively reorganizes spatial information, unleashing the localization potential within CLIP for dense vision-language inference. Furthermore, to enhance the focus on regions of the same categories and local consistency, we propose the Semantic Feedback Refinement (SFR) module, which utilizes semantic segmentation maps to further adjust the attention scores. By integrating these two strategies, our method, termed **ResCLIP**, can be easily incorporated into existing approaches as a plug-and-play module, significantly boosting their performance in dense vision-language inference. Extensive experiments across multiple standard benchmarks demonstrate that our method surpasses state-of-the-art training-free methods, validating the effectiveness of the proposed approach.
For more information, please refer to our [paper](https://arxiv.org/abs/2411.15851).

<p align="center">
  <img src="./figs/method_simplify_all.png" width="800" />
</p>

<p align="center">
  <img src="./figs/pipeline_all.png" width="800" />
</p>

## Main Results

<p align="center">
  <img src="./figs/main_results_wo.png" width="800" />
</p>

<p align="center">
  <img src="./figs/main_results_w.png" width="800" />
</p>

<p align="center">
  <img src="./figs/main_visual.png" width="800" />
</p>


## Getting Started
### Installation

**Step 1: Clone ResCLIP repository:**

```bash
git clone https://github.com/yvhangyang/ResCLIP.git
cd ResCLIP
```

**Step 2: Environment Setup:**

***Create and activate a new conda environment***

```bash
conda create -n ResCLIP
conda activate ResCLIP
```

***Install Dependencies***


```bash
pip install -r requirements.txt
```

### Quick Start

#### Datasets Preparation

Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets including PASCAL VOC, PASCAL Context, Cityscapes, ADE20k, COCO Object and COCO-Stuff164k.
We provide some dataset processing scripts in the `process_dataset.sh`.


####  Evaluation

Before evaluating the model, 

```bash
python eval.py --config ./config/cfg_DATASET.py --workdir YOUR_WORK_DIR
```

or eval on all datasets:
```bash
python eval_all.py
```
Resutls are listed in `YOUR_WORK_DIR/results.txt`.

#### Demo


```bash
python ResCLIP_demo.py
```

## Acknowledgment

This project is based on [NACLIP](https://github.com/sinahmr/NACLIP), [SCLIP](https://github.com/wangf3014/SCLIP), [ClearCLIP](https://github.com/mc-lan/ClearCLIP), [CLIP](https://github.com/openai/CLIP) and [OpenCLIP](https://github.com/mlfoundations/open_clip). Thanks for their excellent works.
