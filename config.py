"""
Configuration settings for the recommendation system
"""

class Config:
    # Paths
    DATA_PATH = "data/ratings_Beauty.csv"
    OUTPUT_DIR = "output/"
    
    # Model parameters
    SVD_COMPONENTS = 10
    SAMPLE_SIZE = 10000
    CORRELATION_THRESHOLD = 0.90
    
    # Visualization settings
    PLOT_STYLE = "ggplot"
    TOP_PRODUCTS_TO_DISPLAY = 30
    
    # Test product (for demonstration)
    DEMO_PRODUCT_ID = "6117036094"
