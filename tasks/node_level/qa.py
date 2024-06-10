from ..base import QATask
from TAGLAS.data import TAGDataset
from torch import LongTensor, Tensor



def default_text_labels(dataset: TAGDataset, split: str, **kwargs) -> tuple[LongTensor, Tensor, list, list, list]:
    r"""Obtain node question answering labels from each dataset for the specified split.
        Each dataset should implement get_NP_indexs_labels and get_NQA_list function.
    Args:
        dataset (TAGDataset): Dataset which implement the get_NP_indexs_labels and get_NQA_list function.
        split (str): Dataset split.
        kwargs: Other arguments.
    """

    sample_indexs, sample_labels, sample_label_maps = dataset.get_NP_indexs_labels(split)
    sample_label_maps, q_list, a_list = dataset.get_NQA_list(sample_label_maps, **kwargs)
    return sample_indexs, sample_labels, sample_label_maps, q_list, a_list


class NQATask(QATask):
    r"""Node-level question answering task. will obtain question and answer list from the dataset.

    """

    def __process_split_and_label__(self):
        sample_indexs, sample_labels, sample_label_maps, q_list, a_list = \
            (default_text_labels(self.dataset, self.split))
        self.question_features = q_list
        self.answer_features = a_list
        return sample_indexs, sample_labels, sample_label_maps
