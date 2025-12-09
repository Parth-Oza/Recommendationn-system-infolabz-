"""
Unit tests for Recommender class
"""

import unittest
import pandas as pd
import numpy as np
from recommender import Recommender

class TestRecommender(unittest.TestCase):
    def setUp(self):
        """Setup test data and model"""
        # Create mock utility matrix
        np.random.seed(42)
        self.X = pd.DataFrame(
            np.random.rand(5, 10),  # 5 products, 10 users
            index=[f'P{i}' for i in range(5)]
        )
        
        # Add some structure for testing correlations
        self.X.loc['P1'] = self.X.loc['P0'] * 0.9 + np.random.rand(10) * 0.1
        self.X.loc['P2'] = self.X.loc['P0'] * 0.8 + np.random.rand(10) * 0.2
        
        self.recommender = Recommender(svd_components=3)
    
    def test_fit(self):
        """Test model fitting"""
        self.recommender.fit(self.X)
        
        # Check that correlation matrix is created
        self.assertIsNotNone(self.recommender.correlation_matrix)
        self.assertIsNotNone(self.recommender.product_names)
        
        # Check shape
        self.assertEqual(
            self.recommender.correlation_matrix.shape, 
            (len(self.X), len(self.X))
        )
    
    def test_recommend(self):
        """Test recommendation generation"""
        self.recommender.fit(self.X)
        recommendations = self.recommender.recommend('P0', n_recommendations=2)
        
        # Should return recommendations
        self.assertTrue(len(recommendations) > 0)
        
        # Should not include the query product
        self.assertNotIn('P0', recommendations)
    
    def test_get_recommendation_scores(self):
        """Test correlation score retrieval"""
        self.recommender.fit(self.X)
        scores = self.recommender.get_recommendation_scores('P0')
        
        # Should return Series with correct length
        self.assertEqual(len(scores), len(self.X))
        
        # P0 should have highest correlation with itself
        self.assertEqual(scores.idxmax(), 'P0')
        self.assertAlmostEqual(scores.max(), 1.0, places=2)
    
    def test_model_not_fitted_error(self):
        """Test error when model not fitted"""
        recommender = Recommender()
        
        with self.assertRaises(ValueError):
            recommender.recommend('P0')
        
        with self.assertRaises(ValueError):
            recommender.get_recommendation_scores('P0')

if __name__ == '__main__':
    unittest.main()
