
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.diagnostic import het_breuschpagan

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from statsmodels.stats.outliers_influence import variance_inflation_factor

import pandas as pd
import numpy as np
import warnings
from collections import Counter
from scipy.stats import shapiro
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import seaborn as sns

class ClassificationOverlay:
    def __init__(self, df, target, visualize=False):
        self.df = df
        self.target = target
        self.visualize = visualize
        self.X = df.drop(columns=[target])
        self.y = df[target]


    def check_separability(self, method="pca", perplexity=30, random_state=1223):
        """
        Projects features into 2D and checks class separability visually and via overlap metrics.

        Parameters:
        - method: "pca" or "tsne"
        - perplexity: t-SNE parameter (ignored for PCA)
        - random_state: reproducibility

        Returns:
        - dict with method used, overlap score, and notes
        """
        X = self.X.values if isinstance(self.X, pd.DataFrame) else self.X
        y = self.y.values if isinstance(self.y, pd.Series) else self.y

        if method == "tsne":
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        else:
            reducer = PCA(n_components=2)

        X_proj = reducer.fit_transform(X)

        # Compute overlap score: mean distance between class centroids vs intra-class spread
        df_proj = pd.DataFrame(X_proj, columns=["dim1", "dim2"])
        df_proj["label"] = y

        centroids = df_proj.groupby("label")[["dim1", "dim2"]].mean()
        spreads = df_proj.groupby("label")[["dim1", "dim2"]].std()

        # Compute pairwise centroid distances
        
        centroid_distances = pdist(centroids.values)
        mean_centroid_dist = np.mean(centroid_distances)
        mean_spread = spreads.values.mean()

        overlap_score = mean_spread / mean_centroid_dist if mean_centroid_dist > 0 else float("inf")
        separable = overlap_score < 0.5  # Tunable threshold

        if self.visualize:
            
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df_proj, x="dim1", y="dim2", hue="label", palette="Set2", alpha=0.7)
            plt.title(f"Class Separability ({method.upper()} projection)")
            plt.tight_layout()
            plt.show()

        return {
            "method": method,
            "overlap_score": round(overlap_score, 3),
            "separable": separable,
            "notes": (
                "Classes appear well-separated in 2D projection."
                if separable else "Significant class overlap — linear models may struggle."
            )
        }
