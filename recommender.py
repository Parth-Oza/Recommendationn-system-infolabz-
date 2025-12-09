"""
Recommendation system implementation using collaborative filtering
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from typing import List, Tuple
from config import Config

class Recommender:
    """Collaborative filtering recommendation system"""
    
    def __init__(self, svd_components: int = None):
        self.svd_components = svd_components or Config.SVD_COMPONENTS
        self.svd = TruncatedSVD(n_components=self.svd_components)
        self.correlation_matrix = None
        self.product_names = None
        
    def fit(self, X: pd.DataFrame) -> 'Recommender':
        """Fit the recommendation model"""
        print("Fitting SVD model...")
        decomposed_matrix = self.svd.fit_transform(X)
        
        print("Computing correlation matrix...")
        self.correlation_matrix = np.corrcoef(decomposed_matrix)
        self.product_names = list(X.index)
        
        return self
    
    def recommend(self, product_id: str, n_recommendations: int = 10) -> List[str]:
        """Get recommendations for a given product"""
        if self.correlation_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Get product index
        product_idx = self.product_names.index(product_id)
        
        # Get correlation scores for all products
        correlation_scores = self.correlation_matrix[product_idx]
        
        # Get indices of highly correlated products
        threshold = Config.CORRELATION_THRESHOLD
        recommended_indices = [
            i for i, score in enumerate(correlation_scores) 
            if score > threshold and self.product_names[i] != product_id
        ]
        
        # Sort by correlation score (descending)
        recommended_indices.sort(
            key=lambda i: correlation_scores[i], 
            reverse=True
        )
        
        # Get top N recommendations
        top_indices = recommended_indices[:n_recommendations]
        recommendations = [self.product_names[i] for i in top_indices]
        
        return recommendations
    
    def get_recommendation_scores(self, product_id: str) -> pd.Series:
        """Get correlation scores for all products with the given product"""
        if self.correlation_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        product_idx = self.product_names.index(product_id)
        correlation_scores = pd.Series(
            self.correlation_matrix[product_idx],
            index=self.product_names
        )
        return correlation_scores.sort_values(ascending=False)
