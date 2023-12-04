# Good Questions Help Zero-Shot Image Reasoning

[Good Questions Help Zero-Shot Image Reasoning](https://arxiv.org/abs/2312.5271167)

QVix leverages LLMs' strong language prior to generate input-exploratory questions with more details than the original query, guiding LVLMs to explore visual content more comprehensively and uncover subtle or peripheral details. QVix enables a wider exploration of visual scenes, improving the LVLMs’ reasoning accuracy and depth in tasks such as visual question answering and visual entailment.

<p align="center" width="100%">
<a ><img src="images/pipeline.svg" alt="overview" style="width: 100%; min-width: 300px; display: block; margin: auto;"></a>
</p>

## Cases

QVix can utilize **detailed information** to better distinguish between options that are easily confused, and achieve a more comprehensive and systematic understanding of images through **contextual information**.

#### Case 1. Detailed Information: Miniature Pinscher vs. Chihuahua 

<p align="center" width="60%">
<a ><img src="images/case1.svg" alt="overview" style="width: 70%; min-width: 300px; display: block; margin: auto;"></a>
</p>

#### Case 2. Contextual Information: Describe the system depicted in the image

<p align="center" width="60%">
<a ><img src="images/case2.svg" alt="overview" style="width: 70%; min-width: 300px; display: block; margin: auto;"></a>
</p>

## Getting Started
**1. Installation**

Git clone our repository and creating conda environment:
```bash
git clone https://github.com/kai-wen-yang/QVix.git
cd QVix
conda create -n QVix python=3.8
conda activate QVix
pip install -r requirement.txt
```
