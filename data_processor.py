"""
Data processing utilities for the recommendation system
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from config import Config

class DataProcessor:
    """Process and prepare data for recommendation system"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or Config.DATA_PATH
        self.df = None
        
    def load_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load and optionally sample the dataset"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.dropna()
        
        if sample_size:
            print(f"Sampling {sample_size} records...")
            self.df = self.df.head(sample_size)
            
        print(f"Dataset shape: {self.df.shape}")
        return self.df
    
    def get_popular_products(self, n: int = 10) -> pd.DataFrame:
        """Get top n most popular products by rating count"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        popular_products = pd.DataFrame(
            self.df.groupby('ProductId')['Rating'].count()
        )
        most_popular = popular_products.sort_values('Rating', ascending=False)
        return most_popular.head(n)
    
    def create_utility_matrix(self) -> pd.DataFrame:
        """Create user-item utility matrix"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        print("Creating utility matrix...")
        utility_matrix = self.df.pivot_table(
            values='Rating', 
            index='UserId', 
            columns='ProductId', 
            fill_value=0
        )
        return utility_matrix
    
    def get_product_index(self, product_id: str, utility_matrix: pd.DataFrame) -> int:
        """Get index of a product in the utility matrix"""
        product_names = list(utility_matrix.columns)
        if product_id not in product_names:
            raise ValueError(f"Product ID {product_id} not found in dataset")
        return product_names.index(product_id)
