"""
Utility functions for visualization and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import pandas as pd
from config import Config

def setup_plotting():
    """Setup matplotlib style and configuration"""
    plt.style.use(Config.PLOT_STYLE)
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12

def plot_popular_products(popular_products: pd.DataFrame, n: int = None):
    """Plot top N most popular products"""
    n = n or Config.TOP_PRODUCTS_TO_DISPLAY
    
    setup_plotting()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot top N products
    data_to_plot = popular_products.head(n)
    bars = ax.bar(
        range(len(data_to_plot)), 
        data_to_plot['Rating'].values,
        color='skyblue',
        edgecolor='darkblue'
    )
    
    # Customize plot
    ax.set_xlabel('Product ID', fontsize=14)
    ax.set_ylabel('Number of Ratings', fontsize=14)
    ax.set_title(f'Top {n} Most Popular Beauty Products', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(data_to_plot)))
    ax.set_xticklabels(data_to_plot.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 5,
            f'{int(height)}',
            ha='center', 
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_recommendations(product_id: str, recommendations: List[str]):
    """Print recommendations in a formatted way"""
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS FOR PRODUCT: {product_id}")
    print(f"{'='*60}")
    
    if not recommendations:
        print("No recommendations found above the threshold.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec}")
    
    print(f"\nTotal recommendations: {len(recommendations)}")
    print(f"Correlation threshold: >{Config.CORRELATION_THRESHOLD}")
