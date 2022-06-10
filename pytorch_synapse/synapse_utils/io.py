import os
import numpy as np
import pandas as pd

from typing import List, Tuple, Optional


def load_features(
        checkpoint_path: str,
        node_idx_list: List[int],
        reload_epoch: int,
        feature_hook: str,
        dataset_path: str,
        l2_normalize: bool,
        contamination_indices_path: Optional[str] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    
    # load feaetures
    features_nf_list = []
    index_n_list = []
    for node_idx in node_idx_list:
        features_nf_list.append(
            np.load(
                os.path.join(
                    checkpoint_path,
                    'features',
                    f'extracted_features__node.{node_idx}__epoch.{reload_epoch}__{feature_hook}.npy')))
        index_n_list.append(
            np.load(
                os.path.join(
                    checkpoint_path,
                    'features',
                    f'extracted_features__node.{node_idx}__epoch.{reload_epoch}__index_array.npy')))
    features_nf = np.concatenate(features_nf_list, axis=0)
    index_n = np.concatenate(index_n_list, axis=0)

    # remove repeat entries (if any; this can happen if the features are extracted in parallel)
    actual_idx_to_cat_idx_map = {actual_idx: cat_idx for cat_idx, actual_idx in enumerate(index_n)}
    cat_keep_indices = list(map(actual_idx_to_cat_idx_map.get, np.unique(index_n)))
    index_n = index_n[cat_keep_indices]
    features_nf = features_nf[cat_keep_indices]

    # sort
    order = np.argsort(index_n)
    index_n = index_n[order]
    features_nf = features_nf[order]

    # l2 normalize?
    if l2_normalize:
        features_nf = features_nf / np.linalg.norm(features_nf, axis=1, keepdims=True)

    # load metadata
    meta_df = pd.read_csv(os.path.join(dataset_path, 'meta.csv'))
    
    # load annotations
    meta_ext_df = pd.read_csv(os.path.join(dataset_path, 'meta_ext.csv'))
    
    # load contamination indices
    if contamination_indices_path is not None:
        contamination_indices = np.load(contamination_indices_path)
        contamination_synapse_ids = set(meta_df.iloc[contamination_indices]['synapse_id'].values)

        # remove contamination from features
        retained_indices = sorted(list(set(np.arange(features_nf.shape[0])).difference(contamination_indices)))
        features_nf = features_nf[retained_indices]
        
        # remove contamination from metadata
        meta_df = meta_df.iloc[retained_indices]
        
        # remove contamination from annotations
        retained_indices_meta_ext = [
            row_idx for row_idx, synapse_id in enumerate(meta_ext_df['synapse_id'].values)
            if synapse_id not in contamination_synapse_ids]
        meta_ext_df = meta_ext_df.iloc[retained_indices_meta_ext]
        
    return features_nf, meta_df, meta_ext_df