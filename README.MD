# <div align="center">RES-RAG: Reinforced Evidence Set Retrieval for Retrieval-Augmented Generation</div>

<div align="center">
<a><img src="https://img.shields.io/badge/language-Python-blue"></a>
<img alt="trl - Version" src="https://img.shields.io/pypi/v/trl?label=trl">
</div>



## Data Preprocessing
We have encapsulated the entire process from HTML to evidence blocks within the [chunking](./chunking/) module. To use it, you can refer to the construct_sft function in [construct_data.py](chunking/construct_data.py), which processes HTML documents to obtain the evidence blocks described in the paper.



## Reinforcement Learning
We use the [TRL](https://github.com/huggingface/trl) framework to deploy the GRPO policy.


## Quick Start
After completing data preprocessing, you can go to [RES-RAG](trl/examples/scripts/resrag.py) to start training GRPO.
After training is complete, you need to merge the trained adapter parameters with the original model. The merging code can be found in [merge_model](./merge_model.py).  Then, choose your preferred method ([vLLM](https://github.com/vllm-project/vllm) or [HuggingFace](https://huggingface.co/)) to launch the model and start inference. 
