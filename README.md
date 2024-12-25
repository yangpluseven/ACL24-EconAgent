# EconAgent with Ollama

Forked from [the official implementation of this ACL 2024 paper](https://github.com/tsinghua-fib-lab/ACL24-EconAgent/tree/master):

Nian Li, Chen Gao, et al. [EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities](https://arxiv.org/abs/2310.10436), ACL 2024.

It is used in the Economics Course (走进经济学) at Ocean University of China for the final assignment of Fall 2024.

# Changes in this fork

Considering that the original implementation is based on chatGPT API, and it is relatively expensive for studnts (who may not be a professional researcher but just a amateur). And the original work used GPT-3.5 API to be specific, which is not in service anymore. So we use Ollama to replace it.

# Run

Install Ollama using one command:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Get llama3.1 model by:

```
ollama run install llama3.1
```

Simulate with Ollama, 100 agents, and 240 months:

`python simulate_ollama.py --num_agents 100 --episode_length 240`

You can modify `--num_agents`, `--episode_length`, `--dialog_len`, `--max_price_inflation`, and `--max_wage_inflation` to change the parameters of the simulation.

You can change the model by modifying `ollama_model` in `simulate_utils.py`.

Generated file will be saved in `data/` directory, you can change it in `simulate_utils.py`.

Pickle file can be loaded as follows:
```
import pickle
file_name = 'Your .pkl file name'
with open(file_name, 'rb') as f:
    data = pickle.load(f)
```
