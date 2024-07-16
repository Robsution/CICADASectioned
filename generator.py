import h5py
import numpy as np
import numpy.typing as npt

from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import data
from typing import List, Tuple


class RegionETGenerator:
    def __init__(
        self, train_size: float = 0.5, val_size: float = 0.1, test_size: float = 0.4
    ):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = 42

    def get_generator(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        batch_size: int,
        drop_reminder: bool = False,
    ) -> data.Dataset:
        dataset = data.Dataset.from_tensor_slices((X, y))
        return (
            dataset.shuffle(210 * batch_size)
            .batch(batch_size, drop_remainder=drop_reminder)
            .prefetch(data.AUTOTUNE)
        )

    def get_data(self, datasets_paths: List[Path]) -> npt.NDArray:
        inputs = []
        for dataset_path in datasets_paths:
            inputs.append(
                h5py.File(dataset_path, "r")["CaloRegions1"][:].astype("float32")
            )
            inputs.append(
                h5py.File(dataset_path, "r")["CaloRegions2"][:].astype("float32")
            )
            inputs.append(
                h5py.File(dataset_path, "r")["CaloRegions3"][:].astype("float32")
            )
        X = np.concatenate(inputs, axis=1)
        X = np.reshape(X, (-1, 18, 14, 1))
        return X

    def get_sectioned_data(self, datasets_paths: List[Path]) -> npt.NDArray:
        inputs = []
        for dataset_path in datasets_paths:
            inputs.append(
                h5py.File(dataset_path, "r")["CaloRegions1"][:].astype("float32")
            )
            inputs.append(
                h5py.File(dataset_path, "r")["CaloRegions2"][:].astype("float32")
            )
            inputs.append(
                h5py.File(dataset_path, "r")["CaloRegions3"][:].astype("float32")
            )
        X = np.swapaxes(np.array(inputs), 0, 1)
        X = np.reshape(X, (-1, 3, 6, 14, 1))
        return X

    def get_data_split(
        self, datasets_paths: List[Path], data_to_use: float
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        X = self.get_data(datasets_paths)
        X_train, X_test = train_test_split(
            X, test_size=self.test_size, random_state=self.random_state
        )
        X_train, X_val = train_test_split(
            X_train,
            test_size=self.val_size / (self.val_size + self.train_size),
            random_state=self.random_state,
        )
        return (X_train[:int(data_to_use*X_train.shape[0])], X_val[:int(data_to_use*X_val.shape[0])], X_test[:int(data_to_use*X_train.shape[0])])

    def get_sectioned_data_split(
        self, datasets_paths: List[Path], data_to_use: float
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        X = self.get_sectioned_data(datasets_paths)
        X_train, X_test = train_test_split(
            X, test_size=self.test_size, random_state=self.random_state
        )
        X_train, X_val = train_test_split(
            X_train,
            test_size=self.val_size / (self.val_size + self.train_size),
            random_state=self.random_state,
        )
        return (X_train[:int(data_to_use*X_train.shape[0])], X_val[:int(data_to_use*X_val.shape[0])], X_test[:int(data_to_use*X_train.shape[0])])

    def get_super_data_split(
        self, datasets_paths: List[Path], data_to_use: float
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        X_train, X_val, X_test = self.get_sectioned_data_split(datasets_paths, data_to_use)
        len_train = X_train.shape[0]
        len_val = X_val.shape[0]
        X_train = np.reshape(X_train, (-1,6,14,1))
        X_val = np.reshape(X_val, (-1,6,14,1))
        np.random.shuffle(X_train)
        np.random.shuffle(X_val)
        return (X_train[:len_train], X_val[:len_val], X_test)
        '''X = self.get_sectioned_data(datasets_paths)
        X_train, X_val, X_test, X_trainval = np.zeros((0,6,14,1)), np.zeros((0,6,14,1)), np.zeros((0,6,14,1)), []
        print(X.shape)
        for i in range(3):
            X_train_temp, X_test_temp = train_test_split(
                X[i], test_size=self.test_size, random_state=self.random_state
            )
            X_trainval.append(np.copy(X_train_temp))
            X_test = np.append(X_test, X_test_temp, axis=0)

        print(X_trainval[0].shape)
        for i in range(3):
            X_train_temp, X_val_temp = train_test_split(
                X_trainval[i], test_size=self.val_size / (self.val_size + self.train_size), random_state=self.random_state
            )
            print(X_train_temp.shape)
            print(X_train.shape)
            X_train = np.append(X_train, X_train_temp, axis=0)
            X_val = np.append(X_val, X_val_temp, axis=0)'''

#        X_train, X_test = train_test_split(
#            X, test_size=self.test_size, random_state=self.random_state
#        )

#        X_train, X_val = train_test_split(
#            X_train,
#            test_size=self.val_size / (self.val_size + self.train_size),
#            random_state=self.random_state,
#        )


        X_train = np.swapaxes(X_train, 0, 1)
        X_train = np.reshape(X, (-1, 3, 6, 14, 1))
        X_val = np.swapaxes(X_val, 0, 1)
        X_val = np.reshape(X, (-1, 3, 6, 14, 1))
        X_test = np.swapaxes(X_test, 0, 1)
        X_test = np.reshape(X, (-1, 3, 6, 14, 1))

        return (X_train, X_val, X_test)

    def get_benchmark(
        self, datasets: dict, filter_acceptance=True
    ) -> Tuple[dict, list]:
        signals = {}
        acceptance = []
        for dataset in datasets:
            if not dataset["use"]:
                continue
            signal_name = dataset["name"]
            for dataset_path in dataset["path"]:
                X = h5py.File(dataset_path, "r")["CaloRegions"][:].astype("float32")
                X = np.reshape(X, (-1, 6, 14, 1))
                try:
                    flags = h5py.File(dataset_path, "r")["AcceptanceFlag"][:].astype(
                        "bool"
                    )
                    fraction = np.round(100 * sum(flags) / len(flags), 2)
                except KeyError:
                    fraction = 100.0
                if filter_acceptance:
                    X = X[flags]
                signals[signal_name] = X
                acceptance.append({"signal": signal_name, "acceptance": fraction})
        return signals, acceptance
