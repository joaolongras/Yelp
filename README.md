# Yelp Restaurant Analytics, Forecasting & Recommendation System

A comprehensive data analysis system for Yelp restaurant reviews that combines sentiment analysis, recommendation algorithms, social network analysis, and time series forecasting to provide deep insights into restaurant performance and customer behavior.

## 📋 Overview

This project analyzes Yelp restaurant data from Sparks city, integrating multiple machine learning and data analysis approaches:

- **Deep Sentiment Analysis**: Uses state-of-the-art transformer models (RoBERTa, BERT) to analyze review sentiment beyond simple star ratings
- **Recommendation Systems**: Implements both User-Based (UBCF) and Item-Based (IBCF) Collaborative Filtering
- **Social Network Analysis**: Explores user relationships and social influence patterns through graph-based methods
- **Time Series Forecasting**: Predicts future review trends using ARIMA models

## 🎯 Key Features

### 1. Sentiment Analysis
- **RoBERTa Model**: `cardiffnlp/twitter-roberta-base-sentiment`
- **BERT Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- Compares sentiment scores with user ratings to identify discrepancies
- Achieves high accuracy in sentiment classification

### 2. Recommendation Systems
- **User-Based Collaborative Filtering (UBCF)**: Recommends restaurants based on similar users' preferences
- **Item-Based Collaborative Filtering (IBCF)**: Recommends restaurants based on item similarity
- Uses cosine similarity for user/item comparison
- Evaluation metrics: RMSE, MAE

### 3. Social Network Analysis
- Constructs directed graphs of user-restaurant-friendship relationships
- Analyzes influence patterns and social propagation
- Computes network metrics:
  - Degree centrality
  - Betweenness centrality
  - Closeness centrality
  - Community detection (Louvain method)
  - Clique analysis
- HITS algorithm for identifying hubs and authorities
- Watts-Strogatz small-world comparison

### 4. Time Series Forecasting
- ARIMA and AutoARIMA models
- Seasonal decomposition
- Stationarity tests (ADF, KPSS)
- 12-month ahead forecasting
- Performance metrics: RMSE, MAE, MAPE, MASE

## 🛠️ Technologies & Libraries

### Core Libraries
- **Data Processing**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **NLP**: `spacy`, `nltk`, `transformers`
- **Deep Learning**: `torch`, `PyTorch`
- **Network Analysis**: `networkx`, `python-louvain`
- **Time Series**: `statsmodels`, `statsforecast`, `pmdarima`
- **Machine Learning**: `scikit-learn`, `surprise`

### Models Used
- `cardiffnlp/twitter-roberta-base-sentiment`
- `nlptown/bert-base-multilingual-uncased-sentiment`
- AutoARIMA for time series prediction

## 📊 Dataset

The project uses the **Yelp Academic Dataset**, focusing on:
- Restaurant businesses in Sparks city
- User reviews and ratings
- User friendship networks
- Temporal review data (2013-2022)

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+
pip install pandas numpy matplotlib seaborn
pip install torch transformers
pip install networkx python-louvain
pip install statsmodels statsforecast pmdarima
pip install scikit-learn scikit-surprise
pip install spacy nltk textblob geopy tqdm
```

### Data Structure

```
data/
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_user.json
├── sparks_reviews.json
├── processed_reviews.csv
├── processed_business.csv
└── processed_users.csv
```

## 📈 Results & Insights

### Sentiment Analysis
- High correlation between sentiment scores and star ratings
- Identifies authentic vs. potentially biased reviews
- RMSE and MAE metrics validate model performance

### Recommendations
- Personalized restaurant suggestions based on user history
- Cold start problem handled through item-based filtering
- Sparse matrix optimization for scalability

### Network Analysis
- Identifies influential users (hubs) in the network
- Detects community structures among users
- Reveals social influence patterns on restaurant choices

### Forecasting
- Predicts future review volume trends
- Captures seasonality in restaurant popularity
- Helps restaurants anticipate demand

## 📁 Project Structure

```
.
├── cac.ipynb                 # Main Jupyter notebook
├── README.md                 # This file
└── data/                     # Data directory
    ├── raw/                  # Original Yelp datasets
    └── processed/            # Preprocessed data
```

## 🔍 Key Analyses

1. **Exploratory Data Analysis (EDA)**
   - Restaurant rating distributions
   - Review length patterns
   - User activity analysis
   - Temporal trends

2. **Sentiment vs. Stars Comparison**
   - Discrepancy analysis
   - Confusion matrices
   - Model accuracy evaluation

3. **Graph Characteristics**
   - Density and diameter
   - Degree distributions
   - Path lengths
   - Homophily measures

4. **Forecast Validation**
   - Train-test split
   - Multiple model comparison
   - Error metrics analysis

## 🎓 Use Cases

- **Restaurant Owners**: Understand customer sentiment and predict future trends
- **Marketing Teams**: Identify influential users and community patterns
- **Recommendation Engines**: Personalized restaurant suggestions
- **Data Scientists**: Comprehensive framework for review analysis

## ⚠️ Limitations

- 2022 data is incomplete due to potential data collection issues
- Cold start problem for new users/restaurants
- Computational intensity for large-scale networks
- Time series predictions depend on historical patterns

## 📝 License

This project uses the Yelp Academic Dataset, subject to Yelp's terms and conditions.

## 👥 Contributors

Project developed as part of academic research in data mining and social network analysis.

## 🔗 References

- Yelp Academic Dataset: https://www.yelp.com/dataset
- RoBERTa: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
- BERT Sentiment: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment

---

**Note**: This is a research/academic project analyzing restaurant reviews and social patterns. Results should be interpreted within the context of the specific dataset used.
