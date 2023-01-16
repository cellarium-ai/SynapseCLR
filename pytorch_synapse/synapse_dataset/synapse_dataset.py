import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from torch.utils.data.dataset import Dataset
from boltons.cacheutils import cachedmethod

_cache = dict()

class SynapseDataset(Dataset):
    
    def __init__(
            self,
            dataset_path: str,
            head: Optional[int] = None,
            supervised_mode: bool = False,
            drop_annotated_synapses: bool = False,
            supervised_column_names: List[str] = [],
            supervised_types: List[str] = [],
            metadata_table: Optional[pd.DataFrame] = None):
        
        self.dataset_path = dataset_path
        self.supervised_mode = supervised_mode
        self.supervised_column_names = supervised_column_names
        self.supervised_types = supervised_types
        
        if supervised_mode:
            assert len(supervised_column_names) == len(supervised_types)
            assert all(entry in {'continuous', 'categorical'} for entry in self.supervised_types)
            # metadata w/ extended annotations
            if metadata_table is not None:
                self.meta_df = metadata_table
            else:
                self.meta_df = pd.read_csv(os.path.join(dataset_path, 'meta_ext.csv'))
            column_names_set = set(self.meta_df.columns.values)
            assert all(column_name in column_names_set for column_name in supervised_column_names)
            self.supervised_column_name_to_type_map = {
                column_name: data_type
                for column_name, data_type in zip(supervised_column_names, supervised_types)}
            
        else:
            # metadata w/o extended annotations
            if metadata_table is not None:
                self.meta_df = metadata_table
            else:
                self.meta_df = pd.read_csv(os.path.join(dataset_path, 'meta.csv'))
        
            if drop_annotated_synapses:
                meta_ext_df = pd.read_csv(os.path.join(dataset_path, 'meta_ext.csv'))
                self.meta_df = self.meta_df[~self.meta_df['synapse_id'].isin(meta_ext_df['synapse_id'])]
                assert len(self.meta_df) > 0
                
        # truncate the dataset to the first `head` entries? (used for debugging)
        if head is not None:
            self.meta_df = self.meta_df.head(head)
        
    @cachedmethod(cache=_cache)
    def get_num_categories(self, column_name: str) -> int:
        assert column_name in self.supervised_column_name_to_type_map
        if self.supervised_column_name_to_type_map[column_name] == 'continuous':
            return 1
        else:
            return len(np.unique(self.meta_df[column_name].values))
    
    @cachedmethod(cache=_cache)
    def get_data_type(self, column_name: str) -> str:
        assert column_name in self.supervised_column_name_to_type_map
        return self.supervised_column_name_to_type_map[column_name]
    
    def __getitem__(self, index: int) -> Union[Tuple[int, np.ndarray], Tuple[int, np.ndarray, np.ndarray]]:
        synapse_meta = self.meta_df.iloc[index]
        if self.supervised_mode:
            annotations_ndarray = np.asarray(
                [synapse_meta[column_name] for column_name in self.supervised_column_names])
            return index, np.load(os.path.join(self.dataset_path, synapse_meta.filename)), annotations_ndarray
        else:
            return index, np.load(os.path.join(self.dataset_path, synapse_meta.filename))

    def __len__(self):
        return len(self.meta_df)
    
    def split(
            self,
            train_fraction: float,
            seed: int) -> Tuple['SynapseDataset', 'SynapseDataset']:
        
        assert 0. < train_fraction < 1.
        
        total_entries = len(self.meta_df)
        train_entries = int(train_fraction * total_entries)
        
        random_indices = np.random.RandomState(seed).permutation(total_entries)
        train_indices = random_indices[:train_entries]
        test_indices = random_indices[train_entries:]
        
        assert len(train_indices) > 0
        assert len(test_indices) > 0
        
        train_meta = self.meta_df.iloc[train_indices].copy().reset_index(drop=True)
        test_meta = self.meta_df.iloc[test_indices].copy().reset_index(drop=True)
        
        train_dataset = SynapseDataset(
            dataset_path=self.dataset_path,
            head=None,
            supervised_mode=self.supervised_mode,
            supervised_column_names=self.supervised_column_names,
            supervised_types=self.supervised_types,
            metadata_table=train_meta)
        
        test_dataset = SynapseDataset(
            dataset_path=self.dataset_path,
            head=None,
            supervised_mode=self.supervised_mode,
            supervised_column_names=self.supervised_column_names,
            supervised_types=self.supervised_types,
            metadata_table=test_meta)
        
        return train_dataset, test_dataset
