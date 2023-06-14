# Graph Instruction Tuning
Recently, large-scale language models (LLMs) shows promising performance on across many NLP tasks, at the same time extending to understand visual modality (Flamingo (2022), LLaVA (2023)). 

However, methods of LLMs to understand structured data (e.g., tabular or graph) modalities have not been actively explored yet. Therefore in this project, I propose Graph Instruction Tuning (GIT), which trains LLMs to understand molecular graph structured data as a reference and output correct answer given natural language instruction (e.g., "Predict the solvation free energies of this molecule in water.", "Is this chemical compound able to cross blood-brain barrier? Answer yes (1) or no (0).")

## Dataset
First, I synthesized natural language instructions corresponding to each dataset using ```openai api```.
Based on manually written seed instruction set, I synthesized 10 instructions for each dataset.

For .json file of git dataset, please run the script below.

```
python construct_dataset.py \
    --bot_model [model version (e.g., gpt-3.5-turbo)] \
    --k [# of instructions per dataset]
    --api_key [your openai api key]
```
A single training instances looks like this.
```
{"smiles":"C(N1CCOCC1)CNC2=NNC(C=C2C)=C3C=CC(=O)C=C3","instruction":"Is this chemical compound able to cross blood-brain barrier? Answer yes (1) or no (0).","y":[[1.0]],"task":"BBBP"}
{"smiles":"C(N1CCOCC1)CNC2=NNC(C=C2C)=C3C=CC(=O)C=C3","instruction":"Please indicate whether this chemical compound is capable of crossing the blood-brain barrier by answering with a \"yes\" (1) or \"no\" (0).","y":[[1.0]],"task":"BBBP"}
```

The created dataset consists of 500K (instruction, molecule smiles, target (label or float)) triplets, which can be used for LLM instruction tuning, thereby making LLM to agent equipped with bio-molecular expertise!

## Plan (Will be updated soon)
I am going to instruction-tune pretrained ```LLaMA-7B``` with 500K ```git_dataset``` synthesized by procedure described above. 