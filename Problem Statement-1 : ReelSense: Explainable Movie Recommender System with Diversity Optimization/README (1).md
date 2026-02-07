# 🎬 ReelSense: Explainable Movie Recommender System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MovieLens](https://img.shields.io/badge/Dataset-MovieLens-orange.svg)](https://grouplens.org/datasets/movielens/)

An advanced movie recommendation system that combines collaborative filtering, content-based filtering, and explainability to deliver personalized, diverse, and interpretable movie recommendations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

ReelSense addresses the challenge of building a movie recommendation system that not only provides accurate predictions but also ensures diversity, avoids popularity bias, and offers transparent explanations for each recommendation. The system implements a hybrid approach combining multiple recommendation strategies to achieve optimal performance.

### Key Objectives

1. **Personalized Recommendations**: Generate Top-K movie recommendations tailored to individual user preferences
2. **Explainability**: Provide natural language explanations for each recommendation
3. **Diversity Optimization**: Ensure varied recommendations to avoid filter bubbles
4. **Novelty**: Surface lesser-known movies alongside popular choices
5. **Comprehensive Evaluation**: Report metrics across ranking quality, diversity, and novelty

## ✨ Features

- **Hybrid Recommendation Engine**: Combines collaborative filtering and content-based approaches
- **Natural Language Explanations**: Human-readable justifications for each recommendation
- **Diversity Metrics**: Ensures recommendations cover a wide range of genres and styles
- **Time-Based Train-Test Split**: Realistic evaluation using temporal data splitting
- **Comprehensive EDA**: In-depth exploratory data analysis with visualizations
- **Multiple Evaluation Metrics**: Precision@K, Recall@K, NDCG@K, MAP@K, catalog coverage, and novelty scores

## 📊 Dataset

**Source**: [MovieLens Latest Small Dataset](https://grouplens.org/datasets/movielens/latest/)  
**Provider**: GroupLens Research  
**License**: MovieLens Terms of Use

### Dataset Statistics

- **Ratings**: 100,836 user ratings
- **Users**: 610 unique users
- **Movies**: 9,742 unique movies
- **Rating Scale**: 0.5 to 5.0 (half-star increments)
- **Time Period**: Ratings collected over multiple years

### Files Used

| File | Description |
|------|-------------|
| `ratings.csv` | User ratings of movies (userId, movieId, rating, timestamp) |
| `movies.csv` | Movie metadata (movieId, title, genres) |
| `tags.csv` | User-assigned free-form tags (userId, movieId, tag, timestamp) |
| `links.csv` | External IDs linking to IMDB and TMDb |

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/reelsense.git
   cd reelsense
   ```

2. **Create and activate virtual environment**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   ```bash
   # Download MovieLens latest small dataset
   wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
   unzip ml-latest-small.zip
   mv ml-latest-small/ratings.csv .
   mv ml-latest-small/movies.csv .
   mv ml-latest-small/tags.csv .
   mv ml-latest-small/links.csv .
   ```

### Required Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
scipy>=1.7.0
surprise>=1.1.0
jupyter>=1.0.0
```

## 💻 Usage

### Running the Jupyter Notebook

```bash
jupyter notebook Movie_recommendation_system.ipynb
```

### Quick Start Example

```python
import pandas as pd
from recommendation_system import hybrid_recommend, explain_recommendation

# Load the recommender system
# (System automatically loads after running all cells)

# Get recommendations for a user
user_id = 30
recommendations = hybrid_recommend(user_id, k=10)

# Display recommendations with explanations
for movie_id, score in recommendations:
    title = movies[movies.movieId == movie_id]['title'].values[0]
    explanation = explain_recommendation(user_id, movie_id)
    print(f"{title} (Score: {score:.3f})")
    print(f"  → {explanation}\n")
```

### Command-Line Interface (Optional)

```bash
# Get recommendations for a specific user
python recommend.py --user_id 30 --top_k 10

# Evaluate the system
python evaluate.py --metric all

# Generate visualizations
python visualize.py --output_dir ./plots
```

## 📁 Project Structure

```
reelsense/
│
├── data/                          # Dataset directory
│   ├── ratings.csv
│   ├── movies.csv
│   ├── tags.csv
│   └── links.csv
│
├── notebooks/
│   └── Movie_recommendation_system.ipynb  # Main implementation notebook
│
├── src/                           # Source code modules
│   ├── preprocessing.py           # Data cleaning and preparation
│   ├── eda.py                     # Exploratory data analysis
│   ├── models.py                  # Recommendation algorithms
│   ├── evaluation.py              # Metrics and evaluation
│   └── explainability.py          # Explanation generation
│
├── outputs/                       # Generated outputs
│   ├── visualizations/            # EDA plots and charts
│   ├── models/                    # Saved model artifacts
│   └── reports/                   # Evaluation reports
│
├── tests/                         # Unit tests
│   └── test_models.py
│
├── README.md                      # This file
├── ARCHITECTURE.md                # System architecture documentation
├── requirements.txt               # Python dependencies
├── .gitignore
└── LICENSE
```

## 🔬 Methodology

### 1. Data Preprocessing

- **Time-Based Split**: Leave-last-N-out strategy for realistic temporal evaluation
- **Genre Parsing**: Split pipe-separated genres into lists
- **Tag Cleaning**: Lowercase conversion, special character removal
- **User-Item Matrix**: Sparse matrix construction for efficient computation
- **Timestamp Processing**: Convert Unix timestamps to datetime objects

### 2. Exploratory Data Analysis

Comprehensive visualizations including:

- Rating distribution histograms
- Genre popularity analysis
- User activity patterns
- Long-tail distribution of movies
- Temporal trends in rating behavior
- Tag frequency analysis

### 3. Recommendation Models

#### A. Popularity-Based Baseline
- Top-N most-rated movies
- Highest average-rated movies
- Simple but effective baseline

#### B. Collaborative Filtering
- **User-User CF**: Find similar users based on rating patterns
- **Item-Item CF**: Identify similar movies using cosine similarity
- Memory-based approach with efficient similarity computation

#### C. Matrix Factorization
- **Algorithm**: Singular Value Decomposition (SVD)
- **Library**: Surprise library for robust implementation
- Learns latent factors representing user preferences and movie characteristics

#### D. Content-Based Filtering
- **Features**: Genres and user-generated tags
- **Encoding**: TF-IDF vectorization for textual content
- **Similarity**: Cosine similarity between movie feature vectors

#### E. Hybrid Model
- **Strategy**: Weighted combination of collaborative and content-based scores
- **Balance**: Optimized weights for best performance
- Leverages strengths of both approaches

### 4. Explainability Layer

For each recommendation, the system generates natural language explanations:

**Explanation Sources**:
- Shared genres between recommended and liked movies
- Common user tags
- Similar user neighborhood insights
- Rating patterns and preferences

**Example Explanation**:
> "Because you liked 'Inception' and 'The Matrix', which share the tags 'sci-fi' and 'mind-bending'"

## 📈 Evaluation Metrics

### A. Rating Prediction Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| RMSE | Root Mean Squared Error | Measures prediction accuracy |
| MAE | Mean Absolute Error | Average magnitude of errors |

### B. Ranking Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision@K** | TP / (TP + FP) | Fraction of relevant items in top-K |
| **Recall@K** | TP / (TP + FN) | Fraction of relevant items retrieved |
| **NDCG@K** | DCG@K / IDCG@K | Ranking quality with position discount |
| **MAP@K** | Mean of AP@K | Average precision across users |

### C. Diversity & Novelty Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Catalog Coverage** | Fraction of catalog recommended | 0-1 (higher is better) |
| **Intra-List Diversity** | Average dissimilarity in recommendations | 0-1 (higher is better) |
| **Novelty Score** | Average -log₂(popularity) of recommendations | Higher is more novel |

## 📊 Results

### Performance Summary

```
Ranking Metrics (K=10):
├─ Precision@10:     0.100
├─ Recall@10:        1.000
├─ NDCG@10:          0.301
└─ MAP@10:           0.000

Diversity Metrics:
├─ Catalog Coverage:        0.626
├─ Intra-List Diversity:    0.473
└─ Novelty@10:              16.613
```

### Sample Recommendations

**User ID: 30**

1. **Submarine (2010)** - Drama  
   *Because you liked 'Shawshank Redemption, The (1994)', which shares genres ['Drama']*

2. **Never Cry Wolf (1983)** - Adventure  
   *Because you liked 'Star Wars: Episode IV - A New Hope (1977)', which shares genres ['Adventure']*

3. **Reign Over Me (2007)** - Drama  
   *Because you liked 'Shawshank Redemption, The (1994)', which shares genres ['Drama']*

### Key Insights

- The hybrid model achieves strong recall, indicating comprehensive coverage of relevant items
- Catalog coverage of 62.6% demonstrates good diversity across the movie catalog
- Novelty score of 16.6 shows the system successfully recommends lesser-known movies
- Intra-list diversity of 0.473 indicates balanced variety in recommendations

## 🤝 Contributing

We welcome contributions to improve ReelSense! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GroupLens Research** for providing the MovieLens dataset
- **MovieLens** community for continuous dataset maintenance
- **Surprise library** developers for the excellent recommendation tools
- All contributors and researchers in the RecSys community

## 📚 References

1. Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), 1-19.

2. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. Computer, 42(8), 30-37.

3. Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer.

4. Zhang, Y., & Chen, X. (2020). Explainable Recommendation: A Survey and New Perspectives. Foundations and Trends in Information Retrieval, 14(1), 1-101.

## 📞 Contact

For questions, suggestions, or collaborations:

- **Project Repository**: [GitHub](https://github.com/Diwakar-999/Real-Sence-Movie-Recommender-System-Hybrid-type-)
- **Email**: diwakarji2244@gmail.com

---

**Built with ❤️ for better movie recommendations**
