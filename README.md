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
<a ><img src="images/case1.png" alt="overview" style="width: 70%; min-width: 300px; display: block; margin: auto;"></a>
</p>

#### Case 2. Contextual Information: Describe the system depicted in the image

<p align="center" width="60%">
<a ><img src="images/case2.png" alt="overview" style="width: 70%; min-width: 300px; display: block; margin: auto;"></a>
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

Add directory to PYTHONPATH:
```bash
cd QVix/models
export PYTHONPATH="$PYTHONPATH:$PWD"
```

**2. Prepare dataset**

You should replace the variable `DATA_DIR` in the `task_datasets/__init__.py` with the directory you save dataset.

SciencQA: in initialing the ScienceQA dataset, the python script will download the test split of ScienceQA from huggingface directly and then saving the samples with image provided DATA_DIR.

**3. Run QVix**

Run QVix on ScienceQA:
```
python tools/eval.py \
--model_name InstructBLIP7B \
--batch_size 4 \
--dataset_name ScienceQA \
--device 0 \
--expname 'QVix' \
--sample_num 1000 \
--task_name vqa_gpt \
--prompt prompt_hand_v1 \
--question 'Question: {}Answer:' \
--api_key
```
```--model_name```: the used LVLM  <br>
```--prompt```: The pre-question generation prompt  <br>
```--api_key```: Your openAI key  <br>
