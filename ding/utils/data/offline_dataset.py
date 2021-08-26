from typing import List, Dict, Optional, Union
import pickle
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class OfflineRLDataset(Dataset):

    def __init__(self, data_path: str) -> None:
        self._data_path = data_path
        # with open(self._data_path, 'rb') as f:
        #     self._data: List[Dict[str, torch.Tensor]] = pickle.load(f)
        with h5py.File(self._data_path, "w") as f:
            to_hdf5(self.__dict__, f)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]

    def load_hdf5(cls, path: str, device: Optional[str] = None) -> "ReplayBuffer":
        """Load replay buffer from HDF5 file."""
        with h5py.File(path, "r") as f:
            buf = cls.__new__(cls)
            buf.__setstate__(from_hdf5(f, device=device))
        return buf


# Note: object is used as a proxy for objects that can be pickled
# Note: mypy does not support cyclic definition currently
Hdf5ConvertibleValues = Union[  # type: ignore
    int, float, Batch, np.ndarray, torch.Tensor, object,
    'Hdf5ConvertibleType',  # type: ignore
]

Hdf5ConvertibleType = Dict[str, Hdf5ConvertibleValues]  # type: ignore


def to_hdf5(x: Hdf5ConvertibleType, y: h5py.Group) -> None:
    """Copy object into HDF5 group."""

    def to_hdf5_via_pickle(x: object, y: h5py.Group, key: str) -> None:
        """Pickle, convert to numpy array and write to HDF5 dataset."""
        data = np.frombuffer(pickle.dumps(x), dtype=np.byte)
        y.create_dataset(key, data=data)

    for k, v in x.items():
        if isinstance(v, (Batch, dict)):
            # dicts and batches are both represented by groups
            subgrp = y.create_group(k)
            if isinstance(v, Batch):
                subgrp_data = v.__getstate__()
                subgrp.attrs["__data_type__"] = "Batch"
            else:
                subgrp_data = v
            to_hdf5(subgrp_data, subgrp)
        elif isinstance(v, torch.Tensor):
            # PyTorch tensors are written to datasets
            y.create_dataset(k, data=to_numpy(v))
            y[k].attrs["__data_type__"] = "Tensor"
        elif isinstance(v, np.ndarray):
            try:
                # NumPy arrays are written to datasets
                y.create_dataset(k, data=v)
                y[k].attrs["__data_type__"] = "ndarray"
            except TypeError:
                # If data type is not supported by HDF5 fall back to pickle.
                # This happens if dtype=object (e.g. due to entries being None)
                # and possibly in other cases like structured arrays.
                try:
                    to_hdf5_via_pickle(v, y, k)
                except Exception as e:
                    raise RuntimeError(
                        f"Attempted to pickle {v.__class__.__name__} due to "
                        "data type not supported by HDF5 and failed."
                    ) from e
                y[k].attrs["__data_type__"] = "pickled_ndarray"
        elif isinstance(v, (int, float)):
            # ints and floats are stored as attributes of groups
            y.attrs[k] = v
        else:  # resort to pickle for any other type of object
            try:
                to_hdf5_via_pickle(v, y, k)
            except Exception as e:
                raise NotImplementedError(
                    f"No conversion to HDF5 for object of type '{type(v)}' "
                    "implemented and fallback to pickle failed."
                ) from e
            y[k].attrs["__data_type__"] = v.__class__.__name__


def from_hdf5(x: h5py.Group, device: Optional[str] = None) -> Hdf5ConvertibleValues:
    """Restore object from HDF5 group."""
    if isinstance(x, h5py.Dataset):
        # handle datasets
        if x.attrs["__data_type__"] == "ndarray":
            return np.array(x)
        elif x.attrs["__data_type__"] == "Tensor":
            return torch.tensor(x, device=device)
        else:
            return pickle.loads(x[()])
    else:
        # handle groups representing a dict or a Batch
        y = dict(x.attrs.items())
        data_type = y.pop("__data_type__", None)
        for k, v in x.items():
            y[k] = from_hdf5(v, device)
        return Batch(y) if data_type == "Batch" else y