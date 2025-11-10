from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def perform_pca(df_encoded, n_components=3, random_seed=42):
    X = df_encoded.values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    
    # handle duplicates for plotting
    df_rounded = pca_df.round(10)
    duplicates_mask = df_rounded.duplicated(keep=False)
    np.random.seed(random_seed)
    pca_df.loc[duplicates_mask, 'PC3'] += np.array([-0.3,0,0.3])
    
    # loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=df_encoded.columns
    )
    
    return pca_df, loadings, pca.explained_variance_ratio_
