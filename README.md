# Amazon Beauty Products Recommendation System

## Project Overview
This project implements a collaborative filtering recommendation system for Amazon Beauty products using Singular Value Decomposition (SVD). The system analyzes customer ratings and purchase patterns to recommend similar products to users.

## Features
- **Popularity-based Filtering**: Identifies top-rated products based on review counts
- **Model-based Collaborative Filtering**: Uses SVD for dimensionality reduction and pattern recognition
- **Product Recommendations**: Suggests similar products based on purchase history correlations
- **Interactive Visualizations**: Shows most popular products through bar charts

## Dataset
- **Source**: Amazon Beauty Products Dataset (from Kaggle)
- **File**: `ratings_Beauty.csv`
- **Size**: 2,023,070 ratings × 4 columns
- **Columns**: 
  - `UserId`: Unique identifier for users
  - `ProductId`: Unique identifier for products  
  - `Rating`: Rating score (1-5)
  - `Timestamp`: Unix timestamp of rating

## Methodology
1. **Data Preparation**: Clean data and create utility matrix
2. **Popularity Analysis**: Identify top-selling products
3. **Collaborative Filtering**: 
   - Create user-item matrix
   - Apply Truncated SVD for dimensionality reduction
   - Compute correlation matrix
   - Generate recommendations based on item similarity

## Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
