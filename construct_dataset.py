from datasets import load_dataset, Dataset
import openai 
import torch_geometric
import backoff
import random 
from torch_geometric.dataset import MoleculeNet
import multiprocessing
import timeit 

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def sendRequest(content, bot_model, queue):
    out = openai.ChatCompletion.create(
                model=bot_model,
                messages=[
                    {"role": "user",
                    "content": content}
                    ]
                )
    queue.put(out['choices'][0]['message']['content'])

def limitedWait(s, queue):
    start = timeit.default_timer()
    while timeit.default_timer() - start < s and queue.empty():
        continue
    return not queue.empty()

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def getResponse(content, bot_model):
    max_tries = 1
    for i in range(0, max_tries):
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target = sendRequest, args=(content, bot_model, queue,))
        p.start()
    return queue.get()


def create_instruction(seed_instructions, bot_model, k):
    for data, inst_li in seed_instructions.items():
        for i in range(k):
            inst = inst_li[i]
            prompt = "Generate the instruction which has same meaning with '{}'.".format(inst)
            inst_n = getResponse(prompt, bot_model)
            seed_instructions[data].append(inst_n)
    return seed_instructions
    
if __name__ == '__main__':

    # define seed instructions of each dataset
    seed_instructions = {
    'ESOL':['Predict the log solubility in mols per liter.'],
    'FreeSolv':['Predict the solvation free energies of this molecule in water.']
    }
    '''
    'HIV':[],
    'BACE':[],
    'BBBP':[],
    'Tox21':[],
    'ClinTox':[],
    'SIDER':[]
    '''
    # create instructions
    instructions = create_instruction(seed_instructions, 'gpt-3.5-turbo', 10)

    # create graph instruction tuning dataset 
    git_data = {}
    git_data['smiles'], git_data['instruction'], git_data['y'], git_data['task'] = [],[],[],[]
    for data in seed_instructions.keys():
        dataset = MoleculeNet(root='', name=data)
        dataset_insts = seed_instructions[data]
        for instance in dataset:
            for inst in dataset_insts:
                git_data['smiles'].append(instance.smiles)
                git_data['instruction'].append(inst) 
                git_data['task'].append(data)
                git_data['y'].append(instance.y)
    
    git_dataset = Dataset.from_dict(git_data)



