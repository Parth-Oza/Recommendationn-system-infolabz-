"""
Unit tests for DataProcessor class
"""

import unittest
import pandas as pd
import numpy as np
from data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """Setup test data"""
        self.processor = DataProcessor()
        # Create mock data
        self.mock_data = pd.DataFrame({
            'UserId': ['A1', 'A1', 'A2', 'A2', 'A3'],
            'ProductId': ['P1', 'P2', 'P1', 'P3', 'P2'],
            'Rating': [5, 4, 3, 5, 2],
            'Timestamp': [1000, 2000, 3000, 4000, 5000]
        })
        self.processor.df = self.mock_data
    
    def test_get_popular_products(self):
        """Test popular products calculation"""
        result = self.processor.get_popular_products(n=2)
        
        # P1 should be most popular (2 ratings)
        self.assertEqual(result.index[0], 'P1')
        self.assertEqual(result.iloc[0]['Rating'], 2)
        
        # Should return top 2
        self.assertEqual(len(result), 2)
    
    def test_create_utility_matrix(self):
        """Test utility matrix creation"""
        matrix = self.processor.create_utility_matrix()
        
        # Check shape
        self.assertEqual(matrix.shape, (3, 3))
        
        # Check specific values
        self.assertEqual(matrix.loc['A1', 'P1'], 5)
        self.assertEqual(matrix.loc['A2', 'P3'], 5)
        
        # Check fill_value
        self.assertEqual(matrix.loc['A1', 'P3'], 0)
    
    def test_get_product_index(self):
        """Test product index retrieval"""
        matrix = self.processor.create_utility_matrix()
        
        idx = self.processor.get_product_index('P2', matrix)
        self.assertEqual(idx, 1)  # P2 is second column
    
    def test_product_index_not_found(self):
        """Test error handling for non-existent product"""
        matrix = self.processor.create_utility_matrix()
        
        with self.assertRaises(ValueError):
            self.processor.get_product_index('P99', matrix)

if __name__ == '__main__':
    unittest.main()
