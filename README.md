# TAGLAS
This repository collect multiple Text-attributed graph (TAG) dataset from multiple source and provide a unified way for preprocessing and loading. 
We also provide a unified task generation pipeline for evaluating the performance of GNN/LLM on these datasets. 
## Statistics
Here are currently support datasets:

| Dataset (key)                 | Avg. #N | Avg. #E  | #G     | Task level | Task                           | Split (train/val/test)    | Domain          | description                                                    | Source                                                                                                                                                    |
|-------------------------------|---------|----------|--------|------------|--------------------------------|---------------------------|-----------------|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cora_node (cora)              | 2708    | 10556    | 1      | Node       | 7-way classification           | 140/500/2068              | Co-Citation     | Predict the category of papers.                                | [Graph-LLM](https://github.com/CurryTang/Graph-LLM), [OFA](https://github.com/LechengKong/OneForAll)                                                      |
| Cora_link (cora)              | 2708    | 10556    | 1      | Link       | Binary classification          | 17944/1056/2112           | Co-Citation     | Predict whether two papers are co-cited by other papers.       | [Graph-LLM](https://github.com/CurryTang/Graph-LLM), [OFA](https://github.com/LechengKong/OneForAll)                                                      |
| Pubmed_node (pubmed)          | 19717   | 88648    | 1      | Node       | 3-way classification           | 60/500/19157              | Co-Citation     | Predict the category of papers.                                | [Graph-LLM](https://github.com/CurryTang/Graph-LLM), [OFA](https://github.com/LechengKong/OneForAll)                                                      |
| Pubmed_link (pubmed)          | 19717   | 88468    | 1      | Link       | Binary classification          | 150700/8866/17730         | Co-Citation     | Predict whether two papers are co-cited by other papers.       | [Graph-LLM](https://github.com/CurryTang/Graph-LLM), [OFA](https://github.com/LechengKong/OneForAll)                                                      |
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
rdkit
```

## Usage
### Load dataset
Use the key of dataset to load the dataset. The key of dataset can be founded in the above table. For example, to load Arxiv dataset:
```python
from TAGLAS import get_datasets
dataset = get_datasets("arxiv")
```
Or, you can load multiple datasets at the same time:
```python
dataset_list = get_datasets(["arxiv", "pcba"])
```
By default, all data are be saved in the `TAGDataset` folder in the root working directory.
If you want to change the data path, you can set the `root` parameter when loading the dataset:
```python
dataset_list = get_datasets(["arxiv", "pcba"], root="your_path")
```
The above function will load dataset in the default way, which is suitable for the most of cases. However, some datasets may have
advanced setting. To have further control over the loading, you can also load dataset by:
```python
from TAGLAS.datasets import FB15K237
fb15k237 = FB15K237(root="your_path", to_undirected=False)
```
### Data key description
The most of datasets contain the following keys:
- `x`: Text feature for all nodes. Usually a `list` or `np.ndarray`.
- `node_map`: A mapping from node index to node text feature. Usually a `torch.LongTensor`.
- `edge_attr`: Text feature for all edges. Usually a `list` or `np.ndarray`.
- `edge_map`: A mapping from edge index to edge text feature. Usually a `torch.LongTensor`.
- `label`: Text feature for all labels. Usually a `list` or `np.ndarray`.
- `label_map`: A mapping from label index to label text feature. Usually a `torch.LongTensor`.
- `edge_index`: The graph structure. Usually a `torch.LongTensor`.

Some dataset may also contain:
- `x_original`: The vector feature for all nodes in the original data source. Usually a `torch.Tensor`.
- `edge_attr_orignal`: The vector feature for all edges in the original data source. Usually a `torch.Tensor`.

To get a specific key:
```python
dataset = get_datasets("arxiv")
x = dataset.x
```

All data samples are stored in the dataset with class 'TAGData', which is inherited from 'Data' in 'torch_geometric' package. To get a single graph sample:
```python
dataset = get_datasets("arxiv")
data = dataset[0]
```
For graph-level datasets, each `data` sample will only contains mapping key like `node_map` or `edge_map` but no real text features. The reason is that,
to minimize the memory usage and avoid saving repeat node/edge text features, all text features are stored in a mapping style. To get the text features, you can use the mapping key to get the text features:
```python
# To get the edge text features for data 0.
data = dataset[0]
edge_attr = dataset.edge_attr[data.edge_map]
# To get the label text features for data 0.
label = dataset.label[data.label_map]

```
You can also get the text features directly from the dataset by calling `get` function:
```python
data = dataset.get(0)
edge_attr = data.edge_attr
```
Notes that for node-level and link-level datasets, the text features are also included in the data sample, as these datasets only contains a single graph.



### Load tasks
In this repository, we provide a unified way to generate tasks based on datasets. Currently, we support the following tasks:
For node-level datasets:
- `DefaultNPTask`: The default node prediction task which return the whole graph as a single sample and use the original node/edge/label features. Usually used for running baseline GNN.
- `SubgraphNPTask`: Extract subgraph for each data sample and use the original node/edge/label features.
- `SubgraphTextNPTask`: Extract subgraph for each data sample and use the text node/edge/label features.
- `NQATask`: Extract subgraph for each data sample and convert the task to question answering format.

For link-level datasets:
- `DefaultLPTask`: The default link prediction task which return the whole graph as a single sample and use the original node/edge/label features. . Usually used for running baseline GNN.
- `SubgraphLPTask`: Extract subgraph for each data sample and use the original node/edge/label features.
- `SubgraphTextLPTask`: Extract subgraph for each data sample and use the text node/edge/label features.
- `LQATask`: Extract subgraph for each data sample and convert the task to question answering format.

For graph-level datasets:
- `DefaultGPTask`: The default graph prediction task using the original node/edge/label features. Usually used for running baseline GNN.
- `DefaultTextGPTask`: The default graph prediction task using the text node/edge/label features.
- `GQATask`: Convert the graph prediction task to question answering format.


