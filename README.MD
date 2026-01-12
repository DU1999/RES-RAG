# <div align="center">RES-RAG: Reinforced Evidence Set Retrieval for Retrieval-Augmented Generation</div>

## Data Preprocessing
We have encapsulated the entire process from HTML to evidence blocks within the chunking module. To use it, you can refer to the construct_sft function in construct_data.py, which processes HTML documents to obtain the evidence blocks described in the paper.
## Reinforcement Learning
We use the TRL framework to deploy the GRPO policy.
## Quick Start
After completing data preprocessing, you can go to trl/examples/scripts/resrag.py to start training GRPO.
After training is complete, you need to merge the trained adapter parameters with the original model. The merging code can be found in X.py.  Then, choose your preferred method (vLLM or Transformers) to launch the model and start inference. 
