# TAGLAS
This repository collects multiple Text-Attributed Graph (TAG) datasets from various sources and 
provides a unified approach for preprocessing and loading. We also offer a standardized task 
generation pipeline for evaluating the performance of GNN/LLM on these datasets. The project is 
still under construction, so please expect more datasets and features in the future. Stay tuned!

## 🔥News
- *2024.06*: First version release.


## Statistics
Here are currently included datasets:

| Dataset (key)                 | Avg. #N | Avg. #E  | #G     | Task level | Task                           | Split (train/val/test)    | Domain          | description                                                    | Source                                                                                                                                                    |
|-------------------------------|---------|----------|--------|------------|--------------------------------|---------------------------|-----------------|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cora_node (cora_node)         | 2708    | 10556    | 1      | Node       | 7-way classification           | 140/500/2068              | Co-Citation     | Predict the category of papers.                                | [Graph-LLM](https://github.com/CurryTang/Graph-LLM), [OFA](https://github.com/LechengKong/OneForAll)                                                      |
| Cora_link (cora_link)         | 2708    | 10556    | 1      | Link       | Binary classification          | 17944/1056/2112           | Co-Citation     | Predict whether two papers are co-cited by other papers.       | [Graph-LLM](https://github.com/CurryTang/Graph-LLM), [OFA](https://github.com/LechengKong/OneForAll)                                                      |
| Pubmed_node (pubmed_node)     | 19717   | 88648    | 1      | Node       | 3-way classification           | 60/500/19157              | Co-Citation     | Predict the category of papers.                                | [Graph-LLM](https://github.com/CurryTang/Graph-LLM), [OFA](https://github.com/LechengKong/OneForAll)                                                      |
| Pubmed_link (pubmed_link)     | 19717   | 88468    | 1      | Link       | Binary classification          | 150700/8866/17730         | Co-Citation     | Predict whether two papers are co-cited by other papers.       | [Graph-LLM](https://github.com/CurryTang/Graph-LLM), [OFA](https://github.com/LechengKong/OneForAll)                                                      |
| Arxiv (arxiv)                 | 169343  | 1166243  | 1      | Node       | 40-way classification          | 90941/29799/48603         | Citation        | Predict the category of papers.                                | [OGB](https://ogb.stanford.edu/), [OFA](https://github.com/LechengKong/OneForAll)                                                                         |
| WikiCS (wikics)               | 11701   | 216123   | 1      | Node       | 10-way classification          | 580/1769/5847             | Wiki page       | Predict the category of wiki pages.                            | [PyG](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WikiCS.html), [OFA](https://github.com/LechengKong/OneForAll) |
| Product-subset (products)     | 54025   | 144638   | 1      | Node       | 47-way classification          | 14695/1567/36982          | Co-purchase     | Predict the category of products.                              | [TAPE](https://github.com/XiaoxinHe/TAPE)                                                                                                                 |
| FB15K237 (fb15k237)           | 14541   | 310116   | 1      | Link       | 237-way classification         | 272115/17535/20466        | Knowledge graph | Predict the relationship between two entities.                 | [OFA](https://github.com/LechengKong/OneForAll)                                                                                                           |
| WN18RR  (wn18rr)              | 40943   | 93003    | 1      | Link       | 11-way classification          | 86835/3034/3134           | Knowledge graph | Predict the relationship between two entities.                 | [OFA](https://github.com/LechengKong/OneForAll)                                                                                                           |
| MovieLens-1m (ml1m)           | 9923    | 2000418  | 1      | Link       | regression/5-way               | 850177/50011/100021       | Movie rating    | Predict the rating between users and movies.                   | [PyG](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MovieLens1M.html)                                             |
| Chembl_pretrain (chemblpre)   | 25.87   | 55.92    | 365065 | Graph      | 1048-way binary classification | 341952/0/0                | molecular       | Predict the effectiveness of molecule to multiple assays.      | [GIMLET](https://github.com/zhao-ht/GIMLET), [OFA](https://github.com/LechengKong/OneForAll)                                                              |
| PCBA (pcba)                   | 25.97   | 56.20    | 437929 | Graph      | 128-way binary classification  | 349854/43650/43588        | molecular       | Predict the effectiveness of molecule to multiple assays.      | [GIMLET](https://github.com/zhao-ht/GIMLET), [OFA](https://github.com/LechengKong/OneForAll)                                                              |
| HIV  (hiv)                    | 25.51   | 54.94    | 41127  | Graph      | Binary classification          | 32901/4113/4113           | molecular       | Predict the effectiveness of molecule to hiv.                  | [GIMLET](https://github.com/zhao-ht/GIMLET), [OFA](https://github.com/LechengKong/OneForAll)                                                              |
| BBBP  (bbbp)                  | 24.06   | 51.91    | 2039   | Graph      | Binary classification          | 1631/204/204              | molecular       | Predict the effectiveness of molecule to brain blood barrier.  | [GIMLET](https://github.com/zhao-ht/GIMLET), [OFA](https://github.com/LechengKong/OneForAll)                                                              |
| BACE  (bace)                  | 34.09   | 73.72    | 1513   | Graph      | Binary classification          | 1210/151/152              | molecular       | Predict the effectiveness of molecule to BACE1 protease.       | [GIMLET](https://github.com/zhao-ht/GIMLET), [OFA](https://github.com/LechengKong/OneForAll)                                                              |
| toxcast (toxcast)             | 18.76   | 38.50    | 8575   | Graph      | 588-way binary classification. | 6859/858/858              | molecular       | Predict the effectiveness of molecule to multiple assays.      | [GIMLET](https://github.com/zhao-ht/GIMLET), [OFA](https://github.com/LechengKong/OneForAll)                                                              |
| esol (esol)                   | 13.29   | 27.35    | 1128   | Graph      | Regression                     | 902/113/113               | molecular       | Predict the solubility of the molecule.                        | [GIMLET](https://github.com/zhao-ht/GIMLET), [OFA](https://github.com/LechengKong/OneForAll)                                                              |
| freesolv (freesolv)           | 8.72    | 16.76    | 642    | Graph      | Regression                     | 513/64/65                 | molecular       | Predict the free energy of hydration of the molecule.          | [GIMLET](https://github.com/zhao-ht/GIMLET), [OFA](https://github.com/LechengKong/OneForAll)                                                              |
| lipo (lipo)                   | 27.04   | 59.00    | 4200   | Graph      | Regression                     | 3360/420/420              | molecular       | Predict the lipophilicity of the molecule.                     | [GIMLET](https://github.com/zhao-ht/GIMLET), [OFA](https://github.com/LechengKong/OneForAll)                                                              |
| cyp450 (cyp450)               | 24.52   | 53.02    | 16896  | Graph      | 5-way binary classification    | 13516/1690/1690           | molecular       | Predict the effectiveness of molecule to CYP450 enzyme family. | [GIMLET](https://github.com/zhao-ht/GIMLET), [OFA](https://github.com/LechengKong/OneForAll)                                                              |
| tox21 (tox21)                 | 18.57   | 38.59    | 7831   | Graph      | 12-way binary classification   | 6264/783/784              | molecular       | Predict the effectiveness of molecule to multiple assays.      | [GIMLET](https://github.com/zhao-ht/GIMLET), [OFA](https://github.com/LechengKong/OneForAll)                                                              |
| muv (muv)                     | 24.23   | 52.56    | 93087  | Graph      | 17-way binary classification   | 74469/9309/9309           | molecular       | Predict the effectiveness of molecule to multiple assays.      | [GIMLET](https://github.com/zhao-ht/GIMLET), [OFA](https://github.com/LechengKong/OneForAll)                                                              |
| ExplaGraphs (expla_graph)     | 5.17    | 4.25     | 2766   | Graph      | Question Answering             | 1659/553/554              | Commonsense     | Common sense reasoning.                                        | [G-retriver](https://github.com/XiaoxinHe/G-Retriever/tree/main)                                                                                          |
| SceneGraphs (scene_graph)     | 19.13   | 68.44    | 100000 | Graph      | Question Answering             | 59978/19997/20025         | scene graph     | Scene graph question answering.                                | [G-retriver](https://github.com/XiaoxinHe/G-Retriever/tree/main)                                                                                          |
| MAG240m-subset (mag240m)      | 5875010 | 26434726 | 1      | Node       | 153-way classification         | 900722/63337/63338/132585 | Citation        | Predict the category of papers.                                | [OGB](https://ogb.stanford.edu/)                                                                                                                          |
| Ultrachat200k (ultrachat200k) | 3.72    | 2.72     | 449929 | Graph      | Question Answering             | 400000/20000/29929        | Conversation    | Answer the question given previous conversation.               | [UltraChat200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)                                                                                                                |

## Requirement
```
PyG>=2.3
datasets
torch>=2.0.1
transformers>=4.36.2
huggingface_hub
rdkit
```

## Installation
You can directly clone the repository into your working project by using the following command:
```
git clone https://github.com/JiaruiFeng/TAGLAS.git
```
We will provide a more user-friendly installation method in the future.

## Usage
### Datasets
#### Load datasets
The basic way to load a dataset is by using its key. The dataset key can be found in the table above. For example, to load the Arxiv dataset:
```python
from TAGLAS import get_dataset
dataset = get_dataset("arxiv")
```
You can also load multiple datasets at the same time:
```python
from TAGLAS import get_datasets
dataset_list = get_datasets(["arxiv", "pcba"])
```
By default, all data files are be saved in the `./TAGDataset` directory root in the repository directory.
If you want to change the data path, you can set the `root` parameter when loading the dataset:
```python
from TAGLAS import get_datasets
dataset_list = get_datasets(["arxiv", "pcba"], root="your_path")
```
The above function will load the dataset in the default way, which is suitable for most cases. However, 
some datasets may have additional arguments. To have further control over the loading process, you can 
also load the dataset by directly add additional arguments:
```python
from TAGLAS import get_dataset
dataset = get_dataset("fb15k237", to_undirected=False)
```
Finally, directly import from the dataset class is also supported:
```python
from TAGLAS.datasets import Arxiv
dataset = Arxiv()
```
#### Data key description and basic usage
All data samples are stored in the dataset with class `TAGData`, which is inherited from `Data` class in 
[`torch_geometric`](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/data.py#L471) package. Different information will be stored in different key. Most datasets contain the following keys:
- `x`: Text feature for all nodes. Usually a `list` or `np.ndarray`.
- `node_map`: A mapping from node index to node text feature. Usually a `torch.LongTensor`.
- `edge_attr`: Text feature for all edges. Usually a `list` or `np.ndarray`.
- `edge_map`: A mapping from edge index to edge text feature. Usually a `torch.LongTensor`.
- `label`: Text feature for all labels. Usually a `list` or `np.ndarray`.
- `label_map`: A mapping from label index to label text feature. Usually a `torch.LongTensor`.
- `edge_index`: The graph structure. Usually a `torch.LongTensor`.

Some datasets may also contain:
- `x_original`: The vector feature of all nodes in the original data source. Usually a `torch.Tensor`.
- `edge_attr_orignal`: The vector feature of all edges in the original data source. Usually a `torch.Tensor`.
- `question`: Text feature of question for the QA tasks.
- `question_map`: A mapping from question index to question text feature.
- `answer`: Text feature of answer for the QA tasks.
- `answer_map`: A mapping from answer index to answer text feature.

Here is a simple demonstration:
```python
from TAGLAS import get_dataset
dataset = get_dataset("arxiv")
# Get node text feature for the whole dataset.
x = dataset.x
# Get the first graph sample in the dataset.
data = dataset[0]
# Get edge text feature for the sample.
edge_attr = data.edge_attr
```

#### Feature mapping
For graph-level datasets, all `_map` keys like `node_map` or `edge_map` will store the mapping to the global feature of 
all data sample. The global features can be accessed by:
```python
from TAGLAS import get_dataset
dataset = get_dataset("hiv")
# Get the global node text features.
dataset.x 
# Get the global edge text features.
dataset.edge_attr
```
The feature for a specific sample can be obtained by:
```python
from TAGLAS import get_dataset
dataset = get_dataset("hiv")
# Global node text features
x = dataset.x
data = dataset[0]
# Get node text feature for sample 0 by the global node_map key of the sample 0.
sample_x = [x[i] for i in data.node_map]
# We also provide direct access to the text feature of each sample by:
sample_x = dataset[0].x
```
For node/edge-level datasets, since they contain only one graph, the local map is also the global map, 
and the logic remains the same. The reason we store the features this way is to avoid repeated text features, 
especially for large datasets with only a few unique text features (like molecule datasets).


### Tasks
#### Supported tasks
In this repository, we provide a unified way to generate tasks based on datasets. Currently, we support the following five task types:
- `default`: The `default` task implements the most common methods used in the graph community for node, edge, and graph-level tasks. Specifically, it returns the entire original graph for node and edge-level tasks and an original graph sample for graph-level tasks. Additionally, it uses the node and edge features from the original source if available; otherwise, it generates identical features. This type is mainly used for debugging and baseline evaluation.
- `default_text`: The logic of `default_text` tasks is the same as `default`, except that all features are replaced with text features. Additionally, we support converting all text features to sentence embeddings.
- `subgraph`:The subgraph task converts node and edge-level tasks into subgraph-based tasks. Specifically, for the target node or edge, it will sample a subgraph around the target. Similar to the default task, it uses the original node and edge features.
- `subgraph_text`: The logic of `subgraph_text` tasks is the same as `subgraph` except that all features are replaced with text features.
- `QA`: The QA task converts all tasks into a question-answering format. Each sample will include a question and an answer key. By default, the QA tasks will sample subgraphs for node and edge-level tasks.

#### Load tasks
To load a specific task, simply call:
```python
from TAGLAS import get_task
# Load default node-level task on cora
task = get_task("cora_node", "default")
# Load subgraph_text edge-level task on pubmed and val split
task = get_task("pubmed_link", "subgraph_text", split="val")
```
Similarly, you can load multiple task at the same time:
```python
from TAGLAS import get_tasks
# Load QA tasks on all datasets.
tasks = get_tasks(["cora_node", "arxiv", "wn18rr", "scene_graph"], "QA")
# Specify task type for each dataset.
tasks = get_tasks(["cora_node", "arxiv"], ["QA", "subgraph_text"])
```
By default, all generated tasks will not be saved. For fast loading and repeat experiments, you can save and load the generated tasks by:
```python
from TAGLAS import get_task
# save_data will save the generated task into corresponding folder. load_saved will try to load the saved task first before generate new task.
arxiv_task = get_task("arxiv", "subgraph_text", split="test", save_data=True, load_saved=True)
# In defualt, the saved task file will be named by used important arguments (like split, hop...). You can also specify it by yourself:
arxiv_task = get_task("arxiv", "subgraph_text", split="test", save_data=True, load_saved=True, save_name="your_name")
```
Directly construct task given dataset is also supported:
Finally, directly import from the dataset class is also supported:
```python
from TAGLAS.datasets import Arxiv
from TAGLAS.tasks import SubgraphTextNPTask
dataset = Arxiv()
# Load subgraph_text node-level task on Arxiv dataset.
task = SubgraphTextNPTask(dataset)
```
#### Convert text feature to sentence embedding
For `default_text`, `subgraph_text`, and `QA` task types, we also provide function to convert raw text feature to sentence embedding:
```python
from TAGLAS import get_task
from TAGLAS.tasks.text_encoder import SentenceEncoder
encoder_name = "ST"
encoder = SentenceEncoder(encoder_name)
arxiv_task = get_task("arxiv", "subgraph_text", split="test")
arxiv_task.convert_text_to_embedding(encoder_name, encoder)
```
In TAGLAS, we implement several commonly used LLMs for sentence embedding, including `ST` (Sentence Transformer), 
`BERT` (vanilla BERT), `e5` (E5), `llama2_7b` (Llama2-7b), and `llama2_13b` (Llama2-13b). You can load different 
models by inputting the respective `model_key` into `SentenceEncoder`. Additionally, you can implement your own 
sentence embedding model as long as it has a `__call__` function to convert input text lists into embeddings.

#### Collate
For all tasks in TAGLAS, we provide a unified collcate function. Specifically, call the collate function by:
```python
from TAGLAS import get_task
arxiv_task = get_task("arxiv", "subgraph_text", split="test")
# Call collate function to get a batch of data
batch = arxiv_task.collate([arxiv_task[i] for i in range(16)])
```
The collate function is implemented based on `torch_geometric.loader.dataloader.Collater`. However, there 
is a major difference. For all text feature keys like `x` and `edge_attr`, it only stores the unique text 
features in the batch. Additionally, all `_map` keys store the map from the corresponding unique text 
features in the batch to all elements.
```python
from TAGLAS import get_task
arxiv_task = get_task("arxiv", "subgraph_text", split="test")
batch = arxiv_task.collate([arxiv_task[i] for i in range(16)])
# to get node text features for all nodes in the batch
x = batch.x[batch.node_map]
# to get edge text features for all edges in the batch
edge_attr = batch.edge_attr[batch.edge_map]
```
In this way, the batch data is more memory and computation efficient, as each unique text only needs to be encoded once.

### Evaluation
"For each dataset and task, we provide a default evaluation tool for performance evaluation based on `torchmetric`. 
Specifically, for each dataset, we support two types of evaluation based on its supported task types."
- `default`: Used for all task types except `QA`. It supports evaluation based on tensor output, which is commonly used.
- `QA`: It supports evaluation based on text output.

To get an evaluator for a certain task, simply call:
```python
from TAGLAS import get_evaluator, get_evaluators
# Get default evaluator for cora_node task. metric_name is a string indicate the name of metric.
metric_name, evaluator = get_evaluator("cora_node", "subgraph_text")
# Get QA evaluator for arxiv
metric_name, evaluator = get_evaluator("arxiv", "QA")
# Get evaluator for multiple input tasks.
metric_name_list, evaluator_list = get_evaluators(["cora_node", "arxiv"], "QA")
```
## Issues and Bugs
The project is still in development. If you encounter any issues or bugs while using it, please feel free to open an issue in the GitHub repository.