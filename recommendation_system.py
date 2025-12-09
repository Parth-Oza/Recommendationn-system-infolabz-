"""
Main recommendation system script
"""

import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from recommender import Recommender
from utils import plot_popular_products, print_recommendations
from config import Config

def main():
    """Main execution function"""
    print("="*60)
    print("AMAZON BEAUTY PRODUCTS RECOMMENDATION SYSTEM")
    print("="*60)
    
    # 1. Load and process data
    processor = DataProcessor()
    df = processor.load_data(sample_size=Config.SAMPLE_SIZE)
    
    print("\n" + "="*60)
    print("DATA ANALYSIS")
    print("="*60)
    print(f"Total records: {len(df):,}")
    print(f"Unique users: {df['UserId'].nunique():,}")
    print(f"Unique products: {df['ProductId'].nunique():,}")
    print(f"Average rating: {df['Rating'].mean():.2f}")
    
    # 2. Get popular products
    print("\n" + "="*60)
    print("POPULAR PRODUCTS ANALYSIS")
    print("="*60)
    
    popular_products = processor.get_popular_products(n=10)
    print("\nTop 10 Most Popular Products:")
    print(popular_products)
    
    # Plot popular products
    plot_popular_products(popular_products, n=Config.TOP_PRODUCTS_TO_DISPLAY)
    
    # 3. Create utility matrix
    print("\n" + "="*60)
    print("COLLABORATIVE FILTERING MODEL")
    print("="*60)
    
    utility_matrix = processor.create_utility_matrix()
    print(f"Utility matrix shape: {utility_matrix.shape}")
    
    # Transpose for item-based filtering
    X = utility_matrix.T
    print(f"Transposed matrix shape: {X.shape}")
    
    # 4. Train recommendation model
    recommender = Recommender(svd_components=Config.SVD_COMPONENTS)
    recommender.fit(X)
    
    # 5. Generate recommendations
    demo_product = Config.DEMO_PRODUCT_ID
    
    # Check if demo product exists in our sample
    if demo_product not in X.index:
        # Use a product from our sample
        demo_product = X.index[0]
        print(f"\nNote: Demo product not in sample. Using: {demo_product}")
    
    recommendations = recommender.recommend(
        product_id=demo_product,
        n_recommendations=10
    )
    
    # 6. Display results
    print_recommendations(demo_product, recommendations)
    
    # 7. Get correlation scores
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    correlation_scores = recommender.get_recommendation_scores(demo_product)
    print(f"\nTop 5 correlated products with {demo_product}:")
    print(correlation_scores.head(6))  # Includes the product itself
    
    return {
        'data': df,
        'popular_products': popular_products,
        'recommendations': recommendations,
        'correlation_scores': correlation_scores
    }

if __name__ == "__main__":
    results = main()
