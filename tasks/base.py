import os
import os.path as osp
import random
import types
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy as c
from typing import (
    Union,
    Any,
    Callable,
    overload,
    Optional
)

import numpy as np
import torch
from torch import Tensor, LongTensor
from torch.utils.data import Dataset
from torch_geometric.loader.dataloader import Collater

from TAGLAS.data import TAGDataset, TAGData
from TAGLAS.utils.graph import edge_index_to_sparse_csr, personalized_pagerank
from .process import feature_embedding_process, subgraph_process, value_to_tensor, parallel_build_sample_process


class BaseTask(Dataset, ABC):
    """Base class for generate tasks. It initialize input dataset and do the process to convert it to specific task.
    The task will be saved in data_list, and each sample is an TAGData element in the data_list. It will contains graph structure
    and feature mapping for the corresponding sample. The corresponding feature lists for all mapping will be cached for later use.

    Args:
        datasets (TAGDataset): Dataset used for generating tasks.
        split (str, optional): Dataset split, choose from ("train", "val", "test").
        save_data (bool, optional): If true, will save generated tasks.
        from_saved (bool, optional): If true, try to load saved task from disk if it exists.
        save_name (str, optional): If given, use the given name in saving instead of default one.
        post_funcs (Union[Callable, list[Callable]], optional): User defined post-processing functions.
            All post_funcs must have two inputs: data and task_class. data is a single task sample and task_class will
            directly input the task instance into the post_func such that the post_func and obtain all information in
            the task instance.
        filter_func (Callable, optional): User defined sample filter function.
    """

    def __init__(
            self,
            dataset: TAGDataset,
            split: str = "train",
            save_data: bool = False,
            from_saved: bool = False,
            save_name: Optional[str] = None,
            post_funcs: Optional[Union[Callable, list[Callable]]] = None,
            filter_func: Optional[Callable] = None,
            **kwargs) -> None:
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.save_data = save_data
        self.data = dataset._data
        self.save_name = save_name
        self.base_collater = Collater(None, None)
        self.post_funcs = post_funcs
        self.filter_func = filter_func
        self.root = osp.join(dataset.root, (dataset.sub_name if "sub_name" in dataset.__dict__ else ""), "task")
        self.dataset_name = (dataset.sub_name if "sub_name" in dataset.__dict__ else dataset.name)
        print(f"Start building {self.__class__.__name__} on dataset {self.dataset_name}...")
        flag = False
        if from_saved:
            flag = self.from_saved()
        if flag:
            pass
        else:
            self.sample_indexs = None
            self.sample_labels = None
            self.sample_label_map = None
            self.node_features = None
            self.edge_features = None
            self.label_features = None
            self.question_features = None
            self.answer_features = None
            self.additional_data = (None)
            self.data_list = []
            self.process()
            if self.save_data:
                self.save_task()
        self.__after_process__()
        # remove intermediate data for saving space.
        self.data = None
        self.sample_indexs = None
        self.sample_labels = None
        self.sample_label_map = None
        print(f"Finish building.")

    @property
    def default_save_name(self):
        return "_".join([self.split])

    @property
    def save_name(self):
        return self._save_name

    @save_name.setter
    def save_name(self, value: str):
        if value is None:
            self._save_name = self.default_save_name
        else:
            self._save_name = value

    @property
    def processed_file_names(self):
        return [f"QA_features.pkl", f"data_list.pkl", f"additional_data.pkl"]

    @property
    def processed_dir(self):
        save_name = self.save_name
        return osp.join(self.root, self.__class__.__name__[:-4], save_name)

    def save_task(self):
        print("Save generated task...")
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        torch.save((self.question_features, self.answer_features),
                   osp.join(self.processed_dir, self.processed_file_names[0]), pickle_protocol=4)
        torch.save(self.data_list, osp.join(self.processed_dir, self.processed_file_names[1]), pickle_protocol=4)
        torch.save(self.additional_data, osp.join(self.processed_dir, self.processed_file_names[2]), pickle_protocol=4)

    def from_saved(self):
        qa_feature_path = osp.join(self.processed_dir, self.processed_file_names[0])
        data_list_path = osp.join(self.processed_dir, self.processed_file_names[1])
        additional_data_path = osp.join(self.processed_dir, self.processed_file_names[2])
        if osp.exists(qa_feature_path) and osp.exists(data_list_path) and osp.exists(additional_data_path):
            print("load task from saved file...")
            self.question_features, self.answer_features = torch.load(qa_feature_path)
            self.data_list = torch.load(data_list_path)
            self.additional_data = torch.load(additional_data_path)
            self.__load_features__()
            return True
        else:
            return False

    @abstractmethod
    def __get_node_feature__(self):
        r"""Function to get node feature for each dataset and return node_features.
        """
        NotImplementedError("Subclass should implement __get_node_feature__ function.")

    @abstractmethod
    def __get_edge_feature__(self):
        r"""Function to get edge feature for each dataset and return edge_features.
        """
        NotImplementedError("Subclass should implement __get_edge_feature__ function.")

    @abstractmethod
    def __get_label_feature__(self):
        r"""Function to get label feature for each dataset and return label_features
        (Mainly used for generate llm embedding).
        """
        NotImplementedError("Subclasses should implement ___get_label_feature__ function.")

    @abstractmethod
    def __process_split_and_label__(self):
        r"""Function to process label and data split. It will return sample index list, label list,
        and mapping from sample to label. For question answering datasets, it may handle question and answer list
        caching as well.
        """
        NotImplementedError("Subclass should implement ___process_label__ function.")

    @abstractmethod
    def __process_graph__(
            self,
            index: Union[int, list, Tensor],
            edge_index: LongTensor,
            node_map: LongTensor,
            edge_map: LongTensor):
        r"""Function to process the graph structure of a single sample and return new edge_index and edge/node mapping.
        For example, extract ego-subgraph around index.
        Args:
            index (Union[int, list, Tensor]): Index of the sample target. can be either node index, graph index, or edge index.
            edge_index (LongTensor): Edge index of the whole graph.
            node_map (LongTensor): Input node mapping from node to text/embedding.
            edge_map (LongTensor): Input edge mapping from edge to text/embedding.
        """
        NotImplementedError("Subclass should implement ___process_graph__ function.")

    @abstractmethod
    def __build_sample__(
            self,
            index: Union[int, Tensor, list],
            y: Union[int, float, Tensor],
            label_map: Union[int, LongTensor, tuple],
            edge_index: LongTensor,
            node_map: LongTensor,
            edge_map: LongTensor,
    ):
        r"""Function to process a single sample and return a TAGData instance.
        Args:
            index (Union[int, Tensor, list]): Index of the sample target. can be either node index, graph index, or edge index.
            y (Union[int, float, Tensor]): label of the sample.
            label_map (Union[int, LongTensor, tuple]): mapping from sample to label in the label list of the whole dataset.
            edge_index (LongTensor): Edge index of the whole graph.
            node_map (LongTensor): Input node mapping from node to text/embedding.
            edge_map (LongTensor): Input edge mapping from edge to text/embedding.
        """
        NotImplementedError("Subclass should implement __build_sample__ function.")

    @abstractmethod
    def __build_task__(self):
        r"""Function to create the final task. Return a data list with each element in the list be a data sample.
        """
        raise NotImplementedError("Subclass should implement __build_task__ function.")

    def __before_process__(self):
        r"""Overwrite and implement this function if additional process is needed before any other process.
        """
        pass

    def __after_process__(self):
        r"""Convert all text features to np.array and process any user defined post process function.
        """
        for key in self.__dict__.keys():
            if "_features" in key:
                value = getattr(self, key, None)
                if value is not None:
                    if isinstance(value, list):
                        setattr(self, key, np.array(value, dtype=object))
        if self.filter_func is not None:
            keep_indexs = [self.filter_func(data) for data in self.data_list]
            self.data_list = [self.data_list[i] for i, keep in enumerate(keep_indexs) if keep]


    def __before_build_task__(self):
        """Overwrite and implement this function if additional process is needed before __build_task__ start.
        """
        pass

    def __after_build_task__(self):
        """Overwrite and implement this function if additional process is needed after the task building.
        """
        pass

    def __before_build_dataset__(self, dataset_idx: int):
        """Overwrite and implement this function if additional process is needed before build each dataset.
        Args:
            dataset_idx (int): dataset index.
        """
        pass

    @overload
    def __before_build_dataset__(self, dataset_idx: int, sample_index: int):
        """Overwrite and implement this function if additional process is needed before build each dataset and each sample.
        Args:
            dataset_idx (int): dataset index.
            sample_index (int): sample index.
        """
        pass

    def __load_features__(self):
        self.__after_build_task__()
        self.node_features = self.__get_node_feature__()
        self.edge_features = self.__get_edge_feature__()
        self.label_features = self.__get_label_feature__()

    def process(self) -> None:
        """Construct the whole task on all input datasets.
        """
        self.__before_process__()
        self.sample_indexs, self.sample_labels, self.sample_label_map = self.__process_split_and_label__()
        self.__before_build_task__()
        self.data_list = self.__build_task__()
        self.__load_features__()

    def __getitem__(self, item: int) -> Any:
        data = c(self.data_list[item])
        node_map = data.node_map
        edge_map = data.edge_map
        label_map = data.label_map
        data.x = self.node_features[node_map.numpy()]
        data.label = self.label_features[label_map.numpy()]
        if self.edge_features is not None:
            data.edge_attr = self.edge_features[edge_map.numpy()]
        if self.post_funcs is not None:
            if isinstance(self.post_funcs, types.FunctionType):
                post_funcs = [self.post_funcs]
            else:
                assert isinstance(self.post_funcs, list)
                post_funcs = self.post_funcs
            for post_func in post_funcs:
                data = post_func(data, task_class=self)
        return data

    def batch_unique_feature(self, features: Union[Tensor, np.ndarray, list]):

        if isinstance(features, Tensor):
            # If feature is tensor, don't need to generate unique feature and feature map.
            unique_feature = features
            feature_map = torch.arange(features.size(0), dtype=torch.long).to(unique_feature.device)
        else:
            if isinstance(features, list) and isinstance(features[0], np.ndarray):
                features = np.concatenate(features, axis=0)
            unique_feature, feature_map = np.unique(features, return_inverse=True)
            feature_map = torch.from_numpy(feature_map).long()

        return unique_feature, feature_map

    def collate(self, batch: list[TAGData], remap_keys: list[str] = ["node", "edge", "label"]):
        batch_data = self.base_collater(batch)
        update_dict = {}
        for key in remap_keys:
            if key == "node":
                feature_key = "x"
            elif key == "edge":
                feature_key = "edge_attr"
            else:
                feature_key = key
            if feature_key in batch_data:
                features = getattr(batch_data, feature_key)
                unique_features, features_map = self.batch_unique_feature(features)
                update_dict[f"{key}_map"] = features_map
                update_dict[feature_key] = unique_features
        batch_data.update(update_dict)
        return batch_data

    def __len__(self):
        return len(self.data_list)

    def _infer_num_classes(self, y: Optional[Tensor]) -> int:
        if y is None:
            return 0
        elif y.numel() == y.size(0) and not torch.is_floating_point(y):
            return int(y.max()) + 1
        elif y.numel() == y.size(0) and torch.is_floating_point(y):
            num_classes = torch.unique(y).numel()
            if num_classes > 2:
                warnings.warn("Found floating-point labels while calling "
                              "`BaseTask.num_classes`. Returning the number of "
                              "unique elements. Please make sure that this "
                              "is expected before proceeding.")
            return num_classes
        else:
            return y.size(-1)

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        data_list = self.data_list
        if 'label_map' in data_list[0] and isinstance(data_list[0].label_map, Tensor):
            label_map = torch.cat([data.label_map for data in data_list if 'label_map' in data], dim=0)
        else:
            label_map = torch.as_tensor([data.y for data in data_list if 'label_map' in data])

        if hasattr(self, '_data_list') and self._data_list is not None:
            self._data_list = self.len() * [None]
        return self._infer_num_classes(label_map)


class DefaultTask(BaseTask):
    """Default task constructor for tasks, it returns the original whole graph as a sample (like
        node-level tasks or whole-graph link prediction). Meanwhile, it uses original node and edge features.
        Mainly used for running baseline and debugging.
    Args:
        datasets (list[TAGDataset]): Dataset used for generating tasks.
        split (str, optional): Dataset split, choose from ("train", "val", "test").
        save_data (bool, optional): If true, will save generated tasks.
        from_saved (bool, optional): If True, try to load saved task instead of regeneration.
        save_name (str, optional): If given, use the given name in saving instead of the default one.
        post_funcs (Union[Callable, list[Callable]], optional): User defined post-processing functions.
            All post_funcs must have two inputs: data and task_class. data is a single task sample and task_class will
            directly input the task instance into the post_func such that the post_func and obtain all information in
            the task instance.
        filter_func (Callable, optional): User defined sample filter function.
        sampling_size (Union[float, int, list], optional): If specified, will do the sampling before generate task.
            if given a float value and value < 1, do the down-sampling, otherwise, do the up-sampling. If value equal to 1.0,
            no sampling is performed. If given a int value, sample exact number of samples.
            User can also directly input a list of index for sampling.
        sampling_mode (str, optional): The sampling method when doing the sampling, choose from random, stratify, and balance.
    """

    def __init__(
            self,
            dataset: TAGDataset,
            split: str = "train",
            save_data: bool = False,
            from_saved: bool = False,
            save_name: Optional[str] = None,
            post_funcs: Optional[Union[Callable, list[Callable]]] = None,
            filter_func: Optional[Callable] = None,
            sample_size: Union[float, int, list] = 1.0,
            sample_mode: str = "random",
            **kwargs) -> None:
        self.sample_size = sample_size
        self.sample_mode = sample_mode
        super().__init__(dataset, split, save_data, from_saved, save_name, post_funcs, filter_func, **kwargs)

    @property
    def default_save_name(self):
        if isinstance(self.sample_size, list):
            return "_".join([self.split, "specified_index"])
        else:
            return "_".join([self.split, str(self.sample_size), self.sample_mode])

    def __sample_slice__(self, content: Union[Tensor, np.ndarray, list], selected_indexs: list) -> Union[
        Tensor, np.ndarray, list]:
        if isinstance(content, Tensor) or isinstance(content, np.ndarray):
            return content[selected_indexs]
        elif isinstance(content, list):
            return np.array(content)[selected_indexs].tolist()
        else:
            raise ValueError("input type not supported.")

    def __random_sampling__(self, num_samples: int, num_selected_samples: int) -> LongTensor:
        selected_indexs = random.choices(list(range(num_samples)), k=num_selected_samples)
        return selected_indexs

    def __balanced_sampling__(self, sample_label_map: list, num_unique_label: int, num_selected_samples: int) -> list:
        selected_indexs = []
        label_dict = {}
        for idx, lb in enumerate(sample_label_map):
            if lb in label_dict:
                label_dict[lb].append(idx)
            else:
                label_dict[lb] = [idx]
        samples_per_label = num_selected_samples // num_unique_label + 1
        # upsample for items without enough label
        for lb, items in label_dict.items():
            if len(items) >= samples_per_label:
                sampled_idx = random.sample(items, samples_per_label)
            else:
                sampled_idx = random.choices(items, k=samples_per_label)
            selected_indexs.extend(sampled_idx)
        return selected_indexs

    def __stratified_sampling__(self, sample_label_map: list, num_samples: int, num_selected_samples: int) -> list:
        selected_indexs = []
        label_dict = {}
        for idx, lb in enumerate(sample_label_map):
            if lb in label_dict:
                label_dict[lb].append(idx)
            else:
                label_dict[lb] = [idx]

        for lb, idx_lst in label_dict.items():
            num_lb_sample = len(idx_lst)
            lb_sample_size = int(num_lb_sample / num_samples * num_selected_samples) + 1
            if num_lb_sample >= lb_sample_size:
                selected_indexs.extend(random.sample(idx_lst, k=lb_sample_size))
            else:
                selected_indexs.extend(idx_lst)
                selected_indexs.extend(random.choice(idx_lst) for _ in range(lb_sample_size - num_lb_sample))
        return selected_indexs

    def __sampling__(self, num_samples: int, num_selected_samples: int) -> list:
        sample_label_map = self.sample_label_map
        sample_mode = self.sample_mode
        if sample_mode != "random":
            # handle graph data which label is list of list
            if isinstance(sample_label_map[0], list):
                if len(sample_label_map[0]) == 1:
                    sample_label_map = [lbs[0] for lbs in sample_label_map]
                else:
                    print(f'Contains multiple labels per sample, use randomly sampling instead.')
                    return self.__random_sampling__(num_samples, num_selected_samples)

            label_map_set = set(sample_label_map)
            num_unique_label = len(label_map_set)
            if num_unique_label > num_samples / 2:
                print(f'Probably not the classification task, use randomly sampling instead.')
                return self.__random_sampling__(num_samples, num_selected_samples)

            if sample_mode == "balanced":
                return self.__balanced_sampling__(sample_label_map, num_unique_label, num_selected_samples)
            elif sample_mode == "stratified":
                return self.__stratified_sampling__(sample_label_map, num_samples, num_selected_samples)
            else:
                raise ValueError(f"sample mode {sample_mode} is not supported. Please choose from random, balanced, or stratified.")

        return self.__random_sampling__(num_samples, num_selected_samples)

    def __before_build_task__(self):
        r"""Perform pre-task sampling.
        """
        num_samples = self.sample_indexs.size(0)
        if isinstance(self.sample_size, list):
            selected_indexs = self.sample_size
        else:
            if isinstance(self.sample_size, float):
                if self.sample_size == 1.0:
                    return
                else:
                    num_selected_samples = int(num_samples * self.sample_size)
            elif isinstance(self.sample_size, int):
                num_selected_samples = self.sample_size
            else:
                raise ValueError("Unrecognized sample_size input.")
            selected_indexs = self.__sampling__(num_samples, num_selected_samples)

        self.sample_indexs = self.__sample_slice__(self.sample_indexs, selected_indexs)
        self.sample_labels = self.__sample_slice__(self.sample_labels, selected_indexs)
        self.sample_label_map = self.__sample_slice__(self.sample_label_map, selected_indexs)

    def __before_process__(self) -> None:
        # before process, convert all text keys to list for fast processing and efficient saving.
        self.data = self.data.text_input_to_list()

    def __get_node_feature__(self) -> Union[Tensor, np.ndarray, list]:
        """
        Use x_original as node feature if it exists, else use node map as one-hot feature. The feature type should be
        taken care of when process the raw data.
        """

        node_features = (
            self.data.x_original if "x_original" in self.data else torch.arange(self.data.node_map.max() + 1).long())
        return node_features

    def __get_edge_feature__(self) -> Union[Tensor, np.ndarray, list]:
        """
        Use edge_attr_original as edge feature if it exists, else use edge_attr,
        or return None if both of them do not exist.
        """

        edge_features = self.data.edge_attr_original if "edge_attr_original" in self.data else None
        return edge_features

    def __get_label_feature__(self) -> Union[Tensor, np.ndarray, list, None]:
        label_features = (self.data.label if "label" in self.data else None)
        return label_features

    def __process_graph__(
            self,
            index: Union[int, list, Tensor],
            edge_index: LongTensor,
            node_map: LongTensor,
            edge_map: LongTensor) -> tuple[LongTensor, LongTensor, LongTensor]:
        """For default tasks, return the original full graph.
        """
        return edge_index, node_map, edge_map

    def __build_sample__(
            self,
            index: Union[int, list, Tensor],
            y: Union[int, float, Tensor],
            label_map: Union[int, LongTensor, tuple],
            edge_index: LongTensor,
            node_map: LongTensor,
            edge_map: LongTensor,
    ) -> TAGData:
        edge_index, node_map, edge_map = self.__process_graph__(index, edge_index, node_map, edge_map)

        label_map = value_to_tensor(label_map)
        y = value_to_tensor(y, to_long=False)
        return TAGData(edge_index=edge_index, node_map=node_map, y=y, label_map=label_map, target_index=index,
                       edge_map=edge_map)

    def __before_build_dataset__(self):
        """preprocess dataset to obtain necessary information.
        """
        edge_index = self.data.edge_index
        node_map = self.data.node_map
        edge_map = self.data.edge_map
        return edge_index, node_map, edge_map

    def __build_task__(self):
        data_list = []
        edge_index, node_map, edge_map = self.__before_build_dataset__()
        sample_label_map = self.sample_label_map
        sample_label_map = value_to_tensor(sample_label_map)

        data_list.append(self.__build_sample__(self.sample_indexs, self.sample_labels, sample_label_map, edge_index,
                                               node_map, edge_map))

        return data_list


class SubgraphTask(DefaultTask):
    """To construct each sample, sample a subgraph of the target nodes. Used for all node/link-level tasks.
    Args:
        datasets (list[TAGDataset]): Dataset used for generating tasks.
        split (str, optional): Dataset split, choose from ("train", "val", "test").
        save_data (bool, optional): If true, will save generated tasks.
        from_saved (bool, optional): If True, try to load saved task instead of regeneration.
        save_name (str, optional): If given, use the given name in saving instead of the default one.
        post_funcs (Union[Callable, list[Callable]], optional): User defined post-processing functions.
            All post_funcs must have two inputs: data and task_class. data is a single task sample and task_class will
            directly input the task instance into the post_func such that the post_func and obtain all information in
            the task instance.
        filter_func (Callable, optional): User defined sample filter function.
        sampling_size (Union[float, int, list], optional): If specified, will do the sampling before generate task.
            if given a float value and value < 1, do the down-sampling, otherwise, do the up-sampling. If value equal to 1.0,
            no sampling is performed. If given an int value, sample exact number of samples.
            User can also directly input a list of index for sampling.
        sampling_mode (str, optional): The sampling method when doing the sampling, choose from random, stratify, and balance.
        hop (int, optional): Number of hop for extracting subgraph. Default is 3.
        max_nodes_per_hop (int, optional): Maximum number of nodes when sampling each hop. If nodes at the current hop is larger
         than the value, randomly select nodes with the number of value. Default is 5.
        num_workers (int, optional): Number of worker for task generation with multiprocess.
        to_sparse (bool, optional): If true, first convert it to sparse tensor then do the subgraph sampling.
                                Efficient when dealing with large graph.
    """

    def __init__(
            self,
            dataset: TAGDataset,
            split: str = "train",
            save_data: bool = False,
            from_saved: bool = False,
            save_name: Optional[str] = None,
            post_funcs: Optional[Union[Callable, list[Callable]]] = None,
            filter_func: Optional[Callable] = None,
            sample_size: Union[float, int, list] = 1.0,
            sample_mode: str = "random",
            hop: int = 3,
            max_nodes_per_hop: int = 5,
            num_workers: int = 0,
            to_sparse: bool = True,
            use_ppr_sampling: bool = False,
            **kwargs) -> None:
        self.hop = hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.num_workers = num_workers
        self.use_ppr_sampling = use_ppr_sampling
        self.ppr_scores = None
        self.to_sparse = to_sparse
        super().__init__(dataset, split, save_data, from_saved, save_name, post_funcs, filter_func, sample_size, sample_mode,
                         **kwargs)

    @property
    def default_save_name(self):
        if isinstance(self.sample_size, list):
            return "_" + "_".join([self.split, str(self.hop), str(self.max_nodes_per_hop), "specified_index"])
        else:
            return "_" + "_".join(
                [self.split, str(self.hop), str(self.max_nodes_per_hop), str(self.sample_size), self.sample_mode])

    def __process_graph__(
            self,
            index: Union[int, list, Tensor],
            edge_index: LongTensor,
            node_map: LongTensor,
            edge_map: LongTensor) -> tuple[LongTensor, LongTensor, LongTensor, LongTensor]:
        """Extract ego-subgraph around the index.
        """
        return subgraph_process(index, edge_index, node_map, edge_map,
                                self.hop, self.max_nodes_per_hop, to_sparse=self.to_sparse, ppr_scores=self.ppr_scores)

    def __build_sample__(
            self,
            index: Union[int, list, Tensor],
            y: Union[int, float, Tensor,],
            label_map: Union[int, LongTensor, tuple],
            edge_index: LongTensor,
            node_map: LongTensor,
            edge_map: LongTensor,
    ):
        index = value_to_tensor(index)
        edge_index, node_map, edge_map, target_index = self.__process_graph__(index, edge_index, node_map, edge_map)
        target_index = value_to_tensor(target_index)
        label_map = value_to_tensor(label_map)
        y = value_to_tensor(y, to_long=False)
        return TAGData(edge_index=edge_index, node_map=node_map, y=y, label_map=label_map, target_index=target_index,
                       edge_map=edge_map)

    def __before_build_dataset__(self):
        """preprocess dataset to obtain necessary information.
        """
        edge_index = self.data.edge_index
        node_map = self.data.node_map
        edge_map = self.data.edge_map
        if self.to_sparse:
            edge_index = edge_index_to_sparse_csr(edge_index, edge_map)
        if self.use_ppr_sampling:
            print("Compute ppr score.")
            self.ppr_scores = personalized_pagerank(edge_index, len(node_map))
            print(self.ppr_scores)
        return edge_index, node_map, edge_map

    def __build_task__(self):
        # edge_index, node_map, edge_map = self.__before_build_dataset__()
        data_list = parallel_build_sample_process(self)
        return data_list


class TextBase():
    """Common functions used for Text-based tasks.
    """

    def __text_to_embedding__(
            self,
            encoder_name: str,
            text_features: Union[list, np.ndarray],
            name: str,
            encoder: Any = None,
            from_saved: bool = True) -> Tensor:
        """Convert text to embedding. If there is saved embedding, directly load it. Otherwise, use the input encoder
        to generate and save it.
        Args:
            encoder_name (str): Name of the encoder. It is also the key map to the saved embedding.
            Please use different name if the encoder function is changed. Otherwise, it
                will load the previous one stead of generating or load the correct one.
            text_features (list[list, np.ndarray]):  Text features to be embedded.
            name (str): Name of the feature, It is also the key map to the saved embedding.
            encoder (Any, optional): A class with encode function that can convert text to embedding.
            from_saved: (bool, optional): If true, first detect the saved embedding for node, edge, and label.
                question and answer feature are not saved to avoid saving mismatch.
        """
        if name in ["question_features", "answer_features"]:
            file = None
        else:
            file_name = osp.join(encoder_name, f"{name}.pt")
            file = osp.join(self.root, file_name)
        embeddings = feature_embedding_process(text_features, encoder, file, from_saved)
        return embeddings

    def __convert_text_to_embedding__(
            self,
            encoder_name: str,
            encoder: Any = None,
            convert_features: list[str] = ["node", "edge", "label"],
            from_saved: bool = True) -> None:
        """Convert all features in convert_features to embedding.
        Args:
            encoder_name (str): Name of the encoder. It is also the key map to the saved embedding.
                Please use different name if the encoder function is changed. Otherwise, it
                will load the previous one stead of generating or load the correct one.
            encoder (Any, optional): A class with encode function that can convert text to embedding.
            convert_features (list[str], optional): A list of key that will be converted.
                All input keys will be appended with "_features" in the process for final matching.
        """
        avaliable_features = []
        exclude_features = []
        for f in convert_features:
            key = f + "_features"
            if key in self.__dict__:
                avaliable_features.append(f)
            else:
                exclude_features.append(key)

        if len(exclude_features) > 0:
            Warning(",".join(exclude_features) + " not in task class, skip encoding it.")

        for f in avaliable_features:
            key = f + "_features"
            setattr(self, key, self.__text_to_embedding__(encoder_name, getattr(self, key), key, encoder, from_saved))


class DefaultTextTask(DefaultTask, TextBase):
    """Default text-based task. It will use the whole graph as the sample. However, the node/edge/label feature will be raw text
    instead of original embedding.
    """

    def __get_node_feature__(self) -> Union[Tensor, np.ndarray, list, None]:
        node_features = (self.data.x if "x" in self.data else None)
        return node_features

    def __get_edge_feature__(self) -> Union[Tensor, np.ndarray, list, None]:
        edge_features = (self.data.edge_attr if "edge_attr" in self.data else None)
        return edge_features

    def convert_text_to_embedding(
            self,
            encoder_name: str,
            encoder: Any = None,
            convert_features: Optional[list[str]] = ["node", "edge", "label"],
            from_saved: Optional[bool] = True) -> None:
        return self.__convert_text_to_embedding__(encoder_name, encoder, convert_features, from_saved)


class SubgraphTextTask(SubgraphTask, TextBase):
    """Text-based tasks with subgraph extraction for each sample.
    """

    def __get_node_feature__(self) -> Union[Tensor, np.ndarray, list, None]:
        node_features = (self.data.x if "x" in self.data else None)
        return node_features

    def __get_edge_feature__(self) -> Union[Tensor, np.ndarray, list, None]:
        edge_features = (self.data.edge_attr if "edge_attr" in self.data else None)
        return edge_features

    def convert_text_to_embedding(
            self,
            encoder_name: str,
            encoder: Any = None,
            convert_features: list[str] = ["node", "edge", "label"],
            from_saved: bool = True) -> None:
        return self.__convert_text_to_embedding__(encoder_name, encoder, convert_features, from_saved)


class QATask(SubgraphTextTask):
    """Question answering (QA) tasks. it will additionally return question and answer mapping. This class mainly used for
    node/link level QA tasks and it is subgraph-based. In default, QA tasks are text-based.
    All texts can be converted by calling convert_text_to_embedding with desired key list.
    """

    def __sampling__(self, num_samples: int, num_selected_samples: int) -> list:
        sample_label_map = self.sample_label_map
        sample_mode = self.sample_mode
        sample_label_map = [label_map[1] for label_map in sample_label_map]
        if sample_mode != "random":
            # handle graph data which label is list of list
            if isinstance(sample_label_map[1], list):
                if len(sample_label_map[0]) == 1:
                    sample_label_map = [lbs[0] for lbs in sample_label_map]
                else:
                    print(f'Contains multiple labels per sample, use randomly sampling instead.')
                    return self.__random_sampling__(num_samples, num_selected_samples)

            label_map_set = set(sample_label_map)
            num_unique_label = len(label_map_set)
            if num_unique_label > num_samples / 2:
                print(f'Probably not the classification task, use randomly sampling instead.')
                return self.__random_sampling__(num_samples, num_selected_samples)

            if sample_mode == "balanced":
                return self.__balanced_sampling__(sample_label_map, num_unique_label, num_selected_samples)
            elif sample_mode == "stratified":
                return self.__stratified_sampling__(sample_label_map, num_samples, num_selected_samples)
            else:
                raise ValueError(f"sample mode {sample_mode} is not supported. Please choose from random, balanced, or stratified.")

        return self.__random_sampling__(num_samples, num_selected_samples)

    def __build_sample__(
            self,
            index: Union[int, Tensor, list],
            y: Union[int, float, Tensor,],
            label_map: Union[int, LongTensor, tuple],
            edge_index: LongTensor,
            node_map: LongTensor,
            edge_map: LongTensor,
    ):
        index = value_to_tensor(index)
        edge_index, node_map, edge_map, target_index = self.__process_graph__(index, edge_index, node_map, edge_map)
        # the input k index will be the first k nodes in the processed graph.

        target_index = value_to_tensor(target_index)
        question_map, label_map, answer_map = label_map
        label_map = value_to_tensor(label_map)
        question_map = value_to_tensor(question_map)
        answer_map = value_to_tensor(answer_map)
        y = value_to_tensor(y, to_long=False)

        return TAGData(edge_index=edge_index, node_map=node_map, y=y, label_map=label_map, target_index=target_index,
                       edge_map=edge_map, question_map=question_map, answer_map=answer_map)

    def convert_text_to_embedding(
            self,
            encoder_name: str,
            encoder: Any = None,
            convert_features: list[str] = ["node", "edge", "label", "question", "answer"],
            from_saved: bool = True, ) -> None:
        return self.__convert_text_to_embedding__(encoder_name, encoder, convert_features, from_saved)

    def __getitem__(self, item: int) -> Any:
        data = c(self.data_list[item])
        node_map = data.node_map
        edge_map = data.edge_map
        label_map = data.label_map
        question_map = data.question_map
        answer_map = data.answer_map

        data.x = self.node_features[node_map.numpy()]
        data.label = self.label_features[label_map.numpy()]
        if self.edge_features is not None:
            data.edge_attr = self.edge_features[edge_map.numpy()]
        data.question = self.question_features[question_map.numpy()]
        data.answer = self.answer_features[answer_map.numpy()]
        if self.post_funcs is not None:
            if isinstance(self.post_funcs, types.FunctionType):
                post_funcs = [self.post_funcs]
            else:
                assert isinstance(self.post_funcs, list)
                post_funcs = self.post_funcs
            for post_func in post_funcs:
                data = post_func(data, task_class=self)
        return data

    def collate(self, batch: list[TAGData], remap_keys: list[str] = ["node", "edge", "label", "question", "answer"]):
        return super(QATask, self).collate(batch, remap_keys)
