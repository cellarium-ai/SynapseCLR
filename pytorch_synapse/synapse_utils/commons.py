import pandas as pd
import numpy as np
import torch


def log1p_zscore(a):
    a = np.log1p(a)
    m = np.mean(a)
    s = np.std(a)
    return (a - m) / s


def load_imputed_annotations(
        meta_df: pd.DataFrame,
        imputed_cell_types_df_path: str,
        imputed_meta_ext_df_path: str,
        cell_type_strategy: str = 'consensus'):

    # MLE consensus cell types
    consensus_cell_types_df = pd.read_csv(imputed_cell_types_df_path, index_col=0).set_index('synapse_id')

    # imputed annotations
    original_imputed_meta_ext_df = pd.read_csv(imputed_meta_ext_df_path, index_col=0).set_index('synapse_id')

    # assert that rows from all tables correspond to the same synapses
    assert(np.all(meta_df['synapse_id'].values == original_imputed_meta_ext_df.index.values))
    assert(np.all(meta_df['synapse_id'].values == consensus_cell_types_df.index.values))
    imputed_meta_ext_df = meta_df.copy()

    imputed_meta_ext_df['cleft_size_log1p_zscore'] = original_imputed_meta_ext_df['imputed__cleft_size_log1p_zscore__mean'].values
    imputed_meta_ext_df['presyn_soma_dist_log1p_zscore'] = original_imputed_meta_ext_df['imputed__presyn_soma_dist_log1p_zscore__mean'].values
    imputed_meta_ext_df['postsyn_soma_dist_log1p_zscore'] = original_imputed_meta_ext_df['imputed__postsyn_soma_dist_log1p_zscore__mean'].values
    imputed_meta_ext_df['mito_size_pre_vx_log1p_zscore_zi'] = original_imputed_meta_ext_df['imputed__mito_size_pre_vx_log1p_zscore_zi__mean'].values
    imputed_meta_ext_df['mito_size_post_vx_log1p_zscore_zi'] = original_imputed_meta_ext_df['imputed__mito_size_post_vx_log1p_zscore_zi__mean'].values
    imputed_meta_ext_df['has_mito_pre'] = original_imputed_meta_ext_df['imputed__has_mito_pre__class_1'].values > 0.5
    imputed_meta_ext_df['has_mito_post'] = original_imputed_meta_ext_df['imputed__has_mito_post__class_1'].values > 0.5

    if cell_type_strategy == 'consensus':
        imputed_meta_ext_df['pre_cell_type'] = consensus_cell_types_df['pre_cell_type__consensus'].values
        imputed_meta_ext_df['post_cell_type'] = consensus_cell_types_df['post_cell_type__consensus'].values
        
    elif cell_type_strategy == 'single':
        imputed_meta_ext_df['pre_cell_type'] = consensus_cell_types_df['imputed__pre_cell_type__class_1'] > 0.5
        imputed_meta_ext_df['post_cell_type'] = consensus_cell_types_df['imputed__post_cell_type__class_1'] > 0.5
        
    else:
        raise ValueError
        
    # misc.
    imputed_meta_ext_df['pre_synaptic_volume_log1p_zscore'] = log1p_zscore(meta_df['pre_synaptic_volume'].values)
    imputed_meta_ext_df['post_synaptic_volume_log1p_zscore'] = log1p_zscore(meta_df['post_synaptic_volume'].values)

    return imputed_meta_ext_df


def get_normal_sample(loc: np.ndarray, scale: np.ndarray):
    return torch.distributions.Normal(
        loc=torch.tensor(loc),
        scale=torch.tensor(scale)).sample([]).cpu().numpy()


def get_bernoulli_sample(probs: np.ndarray):
    return torch.distributions.Bernoulli(
        probs=torch.tensor(probs)).sample([]).cpu().numpy()


def load_imputed_annotations_stochastic(
        meta_df: pd.DataFrame,
        imputed_cell_types_df_path: str,
        imputed_meta_ext_df_path: str,
        cell_type_strategy: str = 'consensus'):

    # MLE consensus cell types
    consensus_cell_types_df = pd.read_csv(imputed_cell_types_df_path, index_col=0).set_index('synapse_id')

    # imputed annotations
    original_imputed_meta_ext_df = pd.read_csv(imputed_meta_ext_df_path, index_col=0).set_index('synapse_id')

    # assert that rows from all tables correspond to the same synapses
    assert(np.all(meta_df['synapse_id'].values == original_imputed_meta_ext_df.index.values))
    assert(np.all(meta_df['synapse_id'].values == consensus_cell_types_df.index.values))
    imputed_meta_ext_df = meta_df.copy()

    imputed_meta_ext_df['cleft_size_log1p_zscore'] = get_normal_sample(
        original_imputed_meta_ext_df['imputed__cleft_size_log1p_zscore__mean'].values,
        original_imputed_meta_ext_df['imputed__cleft_size_log1p_zscore__std'].values)
    
    imputed_meta_ext_df['presyn_soma_dist_log1p_zscore'] = get_normal_sample(
        original_imputed_meta_ext_df['imputed__presyn_soma_dist_log1p_zscore__mean'].values,
        original_imputed_meta_ext_df['imputed__presyn_soma_dist_log1p_zscore__std'].values)
    
    imputed_meta_ext_df['postsyn_soma_dist_log1p_zscore'] = get_normal_sample(
        original_imputed_meta_ext_df['imputed__postsyn_soma_dist_log1p_zscore__mean'].values,
        original_imputed_meta_ext_df['imputed__postsyn_soma_dist_log1p_zscore__std'].values)
        
    
    imputed_meta_ext_df['mito_size_pre_vx_log1p_zscore_zi'] = get_normal_sample(
        original_imputed_meta_ext_df['imputed__mito_size_pre_vx_log1p_zscore_zi__mean'].values,
        original_imputed_meta_ext_df['imputed__mito_size_pre_vx_log1p_zscore_zi__std'].values)
        
    imputed_meta_ext_df['mito_size_post_vx_log1p_zscore_zi'] = get_normal_sample(
        original_imputed_meta_ext_df['imputed__mito_size_post_vx_log1p_zscore_zi__mean'].values,
        original_imputed_meta_ext_df['imputed__mito_size_post_vx_log1p_zscore_zi__std'].values)

    imputed_meta_ext_df['has_mito_pre'] = get_bernoulli_sample(
        original_imputed_meta_ext_df['imputed__has_mito_pre__class_1'].values)
    imputed_meta_ext_df['has_mito_post'] = get_bernoulli_sample(
        original_imputed_meta_ext_df['imputed__has_mito_post__class_1'].values)

    if cell_type_strategy == 'consensus':
        imputed_meta_ext_df['pre_cell_type'] = consensus_cell_types_df['pre_cell_type__consensus'].values
        imputed_meta_ext_df['post_cell_type'] = consensus_cell_types_df['post_cell_type__consensus'].values
        
    elif cell_type_strategy == 'single':
        imputed_meta_ext_df['pre_cell_type'] = get_bernoulli_sample(
            consensus_cell_types_df['imputed__pre_cell_type__class_1'].values)
        imputed_meta_ext_df['post_cell_type'] = get_bernoulli_sample(
            consensus_cell_types_df['imputed__post_cell_type__class_1'].values)
        
    else:
        raise ValueError

    # misc.
    imputed_meta_ext_df['pre_synaptic_volume_log1p_zscore'] = log1p_zscore(meta_df['pre_synaptic_volume'].values)
    imputed_meta_ext_df['post_synaptic_volume_log1p_zscore'] = log1p_zscore(meta_df['post_synaptic_volume'].values)

    return imputed_meta_ext_df

