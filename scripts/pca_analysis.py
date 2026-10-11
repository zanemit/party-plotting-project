from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def perform_pca(df_encoded):
    X = df_encoded.values
    X_scaled = StandardScaler().fit_transform(X)

    pca_all_comps = PCA()
    X_pca = pca_all_comps.fit_transform(X_scaled)

    # pca = PCA(n_components=n_components)
    X_pca = pca_all_comps.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])],
                          index=df_encoded.index)
    
    # handle duplicates for plotting
    df_rounded = pca_df.round(10)
    duplicates_mask = df_rounded.duplicated(keep=False)
    
    # loadings
    loadings = pd.DataFrame(
        pca_all_comps.components_.T* np.sqrt(pca_all_comps.explained_variance_),
        columns=[f'PC{i+1}' for i in range(pca_all_comps.n_components_)],
        index=df_encoded.columns
    )

    return pca_df, loadings, pca_all_comps.explained_variance_ratio_
