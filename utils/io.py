import os
import os.path as osp
import shutil
import zipfile
from typing import (
    Any,
    Optional,
    Union,
)

import fsspec
import torch
from torch_geometric.data import download_url
from huggingface_hub import hf_hub_download
import io
import shutil
from TAGLAS.constants import ROOT


def torch_safe_save(obj: Any, path: str) -> None:
    if obj is not None:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        with fsspec.open(path, 'wb') as f:
            f.write(buffer.getvalue())


def torch_safe_load(path: str, map_location: Any = None) -> Any:
    if osp.exists(path):
        with fsspec.open(path, 'rb') as f:
            return torch.load(f, map_location)
    return None


def download_google_url(
        id: str,
        folder: str,
        filename: str,
        log: bool = True,
):
    r"""Downloads the content of a Google Drive ID to a specific folder."""
    url = f'https://drive.usercontent.google.com/download?id={id}&confirm=t'
    return download_url(url, folder, log, filename)

def download_hf_file(repo_id,
                     filename,
                     local_dir,
                     subfolder=None,
                     repo_type="dataset",
                     cache_dir=None,
                     ):

    hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=filename, repo_type=repo_type,
                    local_dir=local_dir, local_dir_use_symlinks=False, cache_dir=cache_dir, force_download=True)
    if subfolder is not None:
        shutil.move(osp.join(local_dir, subfolder, filename), osp.join(local_dir, filename))
        shutil.rmtree(osp.join(local_dir, subfolder))
    return osp.join(local_dir, filename)



def maybe_log(path, log=True):
    if log:
        print('Extracting', path)


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def move_files_in_dir(source_dir, target_dir):
    # gather all files
    allfiles = os.listdir(source_dir)

    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = os.path.join(source_dir, f)
        dst_path = os.path.join(target_dir, f)
        shutil.move(src_path, dst_path)

def delete_folder(folder_dir: str):
    if osp.isdir(folder_dir):
        for filename in os.listdir(folder_dir):
            file_path = os.path.join(folder_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        print(f"{folder_dir} not exist.")


def delete_processed_files(root: Optional[str]=ROOT,
                           datasets: Optional[Union[str, list[str]]]=None,
                           delete_processed: Optional[bool]=True,
                           delete_raw: Optional[bool]=False):
    if datasets is None:
        datasets = os.listdir(root)
    else:
        if isinstance(datasets, str):
            datasets = [datasets]

    for dataset in datasets:
        path = osp.join(root, dataset)
        if osp.isdir(path):
            task_path = osp.join(path, "task")
            delete_folder(task_path)
            if delete_processed:
                processed_path = osp.join(path, "processed")
                delete_folder(processed_path)
            if delete_raw:
                raw_path = osp.join(path, "raw")
                delete_folder(raw_path)




