import json
import os.path as osp
from copy import deepcopy as c
from itertools import chain

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from TAGLAS.data import TAGData, BaseDict
from .mol_utils import smiles2graph

NAME_TO_SPLIT = {"chemblpre": "chembl_pretraining",
                 "molproperties": "chembl_pretraining",
                 "pcba": "pcba",
                 "hiv": "hiv",
                 "bbbp": "bbbp",
                 "bace": "bace",
                 "toxcast": "toxcast",
                 "esol": "esol",
                 "freesolv": "freesolv",
                 "lipo": "lipo",
                 "cyp450": "cyp450",
                 "tox21": "tox21",
                 "muv": "muv",
                 }


def get_atomic_name_dict(raw_dir):
    file1 = open(osp.join(raw_dir, "id2element.csv"), "r")
    Lines = file1.readlines()
    chem_dict = {}
    for line in Lines:
        line_split = line.strip().split("\t")
        chem_dict[int(line_split[0])] = line_split[2]
    return chem_dict


def load_label_json(name, raw_dir):
    if name == "chemblpre":
        with open(osp.join(raw_dir, "prompt_pretrain.json"), "rb") as f:
            prompt_text = json.load(f)
        return prompt_text["chembl"]
    else:
        with open(osp.join(raw_dir, "mol_label_desc.json"), "rb") as f:
            prompt_text = json.load(f)
        if name in NAME_TO_SPLIT:
            return prompt_text[NAME_TO_SPLIT[name]]
        else:
            raise NotImplementedError("Molecule dataset " + name + " not implemented.")


def get_question_label_texts(task2index, name):
    if name in ["esol", "freesolv", "lipo"]:
        question_texts = [None] * int(len(task2index))
        label_texts = [None] * int(len(task2index))
        for entry in task2index:
            question_texts[task2index[entry][0]] = task2index[entry][1]
            label_texts[task2index[entry][0]] = task2index[entry][1][0]
    else:
        question_texts = [None] * int(len(task2index))
        label_texts = [[None, None]] * int(len(task2index))
        for entry in task2index:
            question_texts[task2index[entry][0]] = task2index[entry][1]
            label_texts[task2index[entry][0]] = [task2index[entry][1][0] + " The answer is No.",
                                                 task2index[entry][1][0] + " The answer is Yes."]
        label_texts.append(["Unavailable."])
    return question_texts, label_texts


def get_raw_dataset(name, raw_dir):
    print("gen text")
    data = load_dataset("haitengzhao/molecule_property_instruction", split=NAME_TO_SPLIT[name], )
    if name == "molproperties":
        data_dict = {"label": data["label"], "task_index": data["task_index"],
                     "molecule_index": data["molecule_index"], "text": data["text"]}
        pd_data = pd.DataFrame.from_dict(data_dict)
        task_data = pd_data[pd.isna(pd_data["task_index"])]
        task_data["ori_index"] = np.arange(len(task_data))
        group = task_data.groupby("molecule_index")
        index = group.ori_index.first()
        labels = group.label.agg(lambda x: x.str.cat(sep=","))
        question_texts = group.text.agg(lambda x: list(chain.from_iterable(x)))
        mol = [data[i]["graph"] for i in index]
        split = [data[i]["split"] for i in index]
        chem_dict = get_atomic_name_dict(raw_dir)

        processed_label_texts = []
        processed_question_texts = []
        graphs = []
        inc = 0
        max_question = 0
        for i in range(len(mol)):
            cur_label = labels[i]
            cur_label = cur_label.split(",")
            num_label = len(cur_label)
            if num_label > max_question:
                max_question = num_label
            cur_question = question_texts[i]
            label_map = torch.arange(inc, inc + num_label)
            question_map = torch.arange(inc, inc + num_label)

            processed_label_texts.extend(cur_label)
            processed_question_texts.extend(cur_question)
            graph = smiles2graph(mol[i], chem_dict)
            graph["label_map"] = label_map
            graph["question_map"] = question_map
            graph["split"] = split[i]
            graphs.append(graph)

        processed_label_texts = np.array(processed_label_texts, dtype=object)
        processed_question_texts = np.array(processed_question_texts, dtype=object)
        return_label_texts, label_inverse_map = np.unique(processed_label_texts, return_inverse=True)
        return_label_texts = np.concatenate([return_label_texts, np.array(["Unavailable."], dtype=object)])
        label_inverse_map = torch.from_numpy(label_inverse_map).long()
        question_texts, question_inverse_map = np.unique(processed_question_texts, return_inverse=True)
        question_texts = np.concatenate([question_texts, np.array(["Unavailable."], dtype=object)])
        question_inverse_map = torch.from_numpy(question_inverse_map).long()
        for i in range(len(graphs)):
            return_label_map = (torch.ones(max_question) * -1).long()
            return_question_map = (torch.ones(max_question) * -1).long()
            return_label_map[:len(graphs[i]["label_map"])] = label_inverse_map[graphs[i]["label_map"]]
            return_question_map[:len(graphs[i]["question_map"])] = question_inverse_map[graphs[i]["question_map"]]
            graphs[i]["label_map"] = return_label_map
            graphs[i]["question_map"] = return_question_map

    else:
        data_dict = {"label": data["label"], "task_index": data["task_index"],
                     "molecule_index": data["molecule_index"], }
        pd_data = pd.DataFrame.from_dict(data_dict)
        task_data = pd_data[np.logical_not(pd.isna(pd_data["task_index"]))]
        task_data["ori_index"] = np.arange(len(task_data))

        group = task_data.groupby("molecule_index")
        index = group.ori_index.first()
        tasks = group.task_index.agg(lambda x: x.str.cat(sep=","))
        labels = group.label.agg(lambda x: x.str.cat(sep=","))
        mol = [data[i]["graph"] for i in index]
        split = [data[i]["split"] for i in index]

        chem_dict = get_atomic_name_dict(raw_dir)
        label_texts = load_label_json(name, raw_dir)
        task2index = {k: [i, label_texts[k]] for i, k in enumerate(label_texts)}

        question_texts, label_texts = get_question_label_texts(task2index, name)

        graphs = []
        if name in ["esol", "freesolv", "lipo"]:
            target_values = [[v for v in labels.iloc[i].split(",")][0] for i in range(len(mol))]
            return_label_texts = [
                label_texts[0] + " Given the molecule, the predicted value is " + "{0:.2f}".format(float(v)) for v in
                target_values]
            return_label_texts, indexs = np.unique(return_label_texts, return_inverse=True)
            for i in range(len(mol)):
                graph = smiles2graph(mol[i], chem_dict)
                cur_label = indexs[i]
                graph["label_map"] = cur_label
                graph["cum_label_map"] = cur_label
                graph["split"] = split[i]
                graphs.append(graph)
        else:
            num_tasks = len(task2index)
            cum = np.array([2 * i for i in range(num_tasks)])
            return_label_texts = list(chain.from_iterable(label_texts))
            for i in range(len(mol)):
                graph = smiles2graph(mol[i], chem_dict)
                task_lst = [task2index[v][0] for v in tasks.iloc[i].split(",")]
                label_lst = [1 if v == "Yes" else 0 for v in labels.iloc[i].split(",")]
                label_map = np.zeros(num_tasks)
                label_map[:] = -1
                label_map[task_lst] = label_lst
                graph["label_map"] = label_map
                cum_label_map = c(label_map)
                cum_label_map[task_lst] = cum_label_map[task_lst] + cum[task_lst]
                graph["cum_label_map"] = cum_label_map
                graph["split"] = split[i]
                graphs.append(graph)
    return graphs, return_label_texts, question_texts


def gen_graph(graphs, name):
    for g in graphs:
        g["node_feat"] = ["Chemical atom with the following information: " + t for t in g["node_feat"]]
        g["edge_feat"] = ["Chemical bond between two atoms with the following information: " + t for t in g["edge_feat"]]


    node_texts = []
    edge_texts = []
    data_list = []
    for g in graphs:
        node_texts += g["node_feat"]
        edge_texts += g["edge_feat"]
    unique_node_texts = set(node_texts)
    unique_edge_texts = set(edge_texts)
    u_node_texts_lst = list(unique_node_texts)
    u_edge_texts_lst = list(unique_edge_texts)
    node_texts2id = {v: i for i, v in enumerate(u_node_texts_lst)}
    edge_texts2id = {v: i for i, v in enumerate(u_edge_texts_lst)}

    split = BaseDict({"train": [], "valid": [], "test": []})
    for i, g in enumerate(graphs):
        cur_nt_id = [node_texts2id[v] for v in g["node_feat"]]
        cur_et_id = [edge_texts2id[v] for v in g["edge_feat"]]
        if name == "molproperties":
            data_list.append(
                TAGData(node_map=torch.tensor(cur_nt_id, dtype=torch.long),
                        edge_map=torch.tensor(cur_et_id, dtype=torch.long),
                        edge_index=torch.tensor(g["edge_list"], dtype=torch.long).T,
                        label_map=g["label_map"].unsqueeze(0),
                        question_map=g["question_map"].unsqueeze(0)
                        )
            )
        else:
            data_list.append(
                TAGData(node_map=torch.tensor(cur_nt_id, dtype=torch.long),
                        edge_map=torch.tensor(cur_et_id, dtype=torch.long),
                        edge_index=torch.tensor(g["edge_list"], dtype=torch.long).T,
                        label_map=torch.tensor(g["label_map"]).long().unsqueeze(0),
                        cum_label_map=torch.tensor(g["cum_label_map"]).long().unsqueeze(0),
                        )
            )
        split[g["split"]].append(i)

    split.rename_key("val", "valid")

    ret = (data_list, [u_node_texts_lst, u_edge_texts_lst],
           split,)
    return ret
