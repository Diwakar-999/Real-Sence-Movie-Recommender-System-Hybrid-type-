# 🏗️ ReelSense Architecture Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Design](#architecture-design)
3. [Data Flow](#data-flow)
4. [Component Details](#component-details)
5. [Algorithm Implementation](#algorithm-implementation)
6. [Scalability Considerations](#scalability-considerations)
7. [Future Enhancements](#future-enhancements)

---

## 1. System Overview

### 1.1 High-Level Architecture

ReelSense implements a **multi-layered hybrid recommendation system** that combines collaborative filtering, content-based filtering, and explainability modules to generate personalized movie recommendations.

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
│                    (Jupyter Notebook / API)                  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Recommendation Engine                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Popularity │  │ Collaborative│  │  Content-Based   │   │
│  │   Baseline  │  │  Filtering   │  │   Filtering      │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Hybrid Model (Weighted Ensemble)           │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Explainability Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │Genre-based   │  │  Tag-based   │  │  Collaborative   │  │
│  │Explanations  │  │ Explanations │  │  Explanations    │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     Evaluation Module                        │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐    │
│  │ Ranking  │  │Diversity │  │     Novelty            │    │
│  │ Metrics  │  │ Metrics  │  │     Metrics            │    │
│  └──────────┘  └──────────┘  └────────────────────────┘    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Ratings  │  │  Movies  │  │   Tags   │  │  Links   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

- **Modularity**: Each component (collaborative, content-based, explainability) is independent and interchangeable
- **Scalability**: Efficient sparse matrix operations and batch processing
- **Explainability**: Every recommendation comes with human-readable justification
- **Diversity**: Active measures to prevent filter bubbles and popularity bias
- **Extensibility**: Easy to add new recommendation algorithms or features

---

## 2. Architecture Design

### 2.1 Layered Architecture

#### Layer 1: Data Layer
**Purpose**: Data ingestion, storage, and preprocessing

**Components**:
- CSV file readers for MovieLens dataset
- Data validation and cleaning pipelines
- Sparse matrix constructors
- Feature extractors

**Key Responsibilities**:
- Load and validate data from CSV files
- Handle missing values and duplicates
- Parse complex fields (genres, tags, timestamps)
- Create efficient data structures for computation

#### Layer 2: Feature Engineering Layer
**Purpose**: Transform raw data into features for recommendation models

**Components**:
- Genre encoder (MultiLabelBinarizer)
- Tag vectorizer (TF-IDF)
- User-item matrix builder
- Similarity matrix calculator

**Key Responsibilities**:
- Encode categorical variables
- Create content feature vectors
- Build user-item interaction matrices
- Compute similarity matrices (cosine, Pearson)

#### Layer 3: Model Layer
**Purpose**: Core recommendation algorithms

**Components**:
1. **Popularity Model**
   - Most-rated movies ranker
   - Highest-rated movies selector

2. **Collaborative Filtering**
   - User-User CF using k-nearest neighbors
   - Item-Item CF using cosine similarity

3. **Matrix Factorization**
   - SVD implementation via Surprise library
   - Latent factor learning

4. **Content-Based Filtering**
   - Genre-based similarity
   - Tag-based similarity
   - Combined content features

5. **Hybrid Model**
   - Weighted score combination
   - Dynamic weight adjustment

**Key Responsibilities**:
- Generate recommendation scores
- Rank items by predicted preference
- Handle cold-start scenarios
- Optimize for diversity

#### Layer 4: Explainability Layer
**Purpose**: Generate human-readable explanations for recommendations

**Components**:
- Genre overlap analyzer
- Tag similarity matcher
- Collaborative neighbor identifier
- Natural language template engine

**Key Responsibilities**:
- Identify reasons for recommendations
- Format explanations in natural language
- Link recommended items to user history
- Provide transparency in decision-making

#### Layer 5: Evaluation Layer
**Purpose**: Measure system performance across multiple dimensions

**Components**:
- Ranking metrics calculator (Precision, Recall, NDCG, MAP)
- Diversity metrics calculator (Coverage, Intra-list diversity)
- Novelty scorer
- Statistical significance testers

**Key Responsibilities**:
- Evaluate recommendation quality
- Measure diversity and coverage
- Assess novelty of recommendations
- Compare different model variants

---

## 3. Data Flow

### 3.1 Training Pipeline

```
┌─────────────┐
│  Raw Data   │
│  (CSV files)│
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Data Loading    │
│  & Validation    │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Preprocessing   │
│  - Parse genres  │
│  - Clean tags    │
│  - Time split    │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│Feature Engineering│
│  - TF-IDF vectors│
│  - Genre encoding│
│  - Similarity mtx│
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Model Training  │
│  - User-User CF  │
│  - Item-Item CF  │
│  - SVD           │
│  - Content model │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Hybrid Model    │
│  Construction    │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Model Storage   │
│  & Serialization │
└──────────────────┘
```

### 3.2 Inference Pipeline

```
┌─────────────┐
│  User Query │
│  (user_id,k)│
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│  Retrieve User       │
│  History & Profile   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Generate Scores     │
│  (All Models)        │
│  ┌────────────────┐  │
│  │ Collaborative  │  │
│  │ Content-Based  │  │
│  │ Popularity     │  │
│  └────────────────┘  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Hybrid Score        │
│  Combination         │
│  score = α×CF +      │
│          β×Content   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Re-ranking for      │
│  Diversity           │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Top-K Selection     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Generate            │
│  Explanations        │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Return              │
│  Recommendations     │
│  + Explanations      │
└──────────────────────┘
```

---

## 4. Component Details

### 4.1 Data Preprocessing Component

**Input**: Raw CSV files (ratings.csv, movies.csv, tags.csv, links.csv)  
**Output**: Cleaned DataFrames, train/test splits, feature matrices

**Implementation Details**:

```python
# Pseudo-code structure
class DataPreprocessor:
    def __init__(self):
        self.ratings = None
        self.movies = None
        self.tags = None
        
    def load_data(self, data_path):
        """Load all CSV files"""
        pass
        
    def clean_genres(self):
        """Split pipe-separated genres into lists"""
        # movies['genres'] = movies['genres'].str.split('|')
        pass
        
    def clean_tags(self):
        """Lowercase and remove special characters"""
        pass
        
    def temporal_split(self, n_last=1):
        """Leave-last-N-out split per user"""
        pass
        
    def build_user_item_matrix(self):
        """Create sparse interaction matrix"""
        pass
```

**Key Functions**:
- `temporal_split()`: Splits data chronologically, keeping last N interactions per user for testing
- `build_user_item_matrix()`: Creates sparse CSR matrix for efficient computation
- `clean_tags()`: Normalizes user-generated tags for consistent matching

### 4.2 Collaborative Filtering Component

**Approach**: Memory-based collaborative filtering using similarity matrices

**User-User Collaborative Filtering**:

```python
# Pseudo-code structure
class UserUserCF:
    def __init__(self, k_neighbors=20):
        self.k = k_neighbors
        self.user_similarity = None
        
    def fit(self, user_item_matrix):
        """Compute user-user similarity matrix"""
        # Cosine similarity between user vectors
        self.user_similarity = cosine_similarity(user_item_matrix)
        
    def predict(self, user_id, movie_id):
        """Predict rating based on similar users"""
        # Find k most similar users who rated the movie
        # Weighted average of their ratings
        pass
        
    def recommend(self, user_id, k=10):
        """Generate top-k recommendations"""
        # Score all unrated movies
        # Return top-k by predicted rating
        pass
```

**Item-Item Collaborative Filtering**:

```python
class ItemItemCF:
    def __init__(self):
        self.item_similarity = None
        
    def fit(self, user_item_matrix):
        """Compute item-item similarity matrix"""
        # Cosine similarity between movie vectors
        self.item_similarity = cosine_similarity(user_item_matrix.T)
        
    def recommend(self, user_id, user_history, k=10):
        """Generate recommendations based on similar items"""
        # For each movie user liked
        #   Find similar movies
        #   Aggregate scores
        # Return top-k
        pass
```

**Complexity Analysis**:
- Similarity matrix computation: O(n² × m) where n = users/items, m = avg interactions
- Recommendation generation: O(k × m) where k = neighbors, m = movies

### 4.3 Content-Based Filtering Component

**Features Used**:
1. Genres (multi-label binary encoding)
2. User tags (TF-IDF vectorization)
3. Combined content vector

**Implementation**:

```python
class ContentBasedFiltering:
    def __init__(self):
        self.genre_features = None
        self.tag_features = None
        self.content_similarity = None
        
    def encode_genres(self, movies_df):
        """One-hot encode genres"""
        mlb = MultiLabelBinarizer()
        genres_list = movies_df['genres'].str.split('|')
        self.genre_features = mlb.fit_transform(genres_list)
        
    def vectorize_tags(self, tags_df):
        """TF-IDF vectorization of tags"""
        tfidf = TfidfVectorizer(max_features=100)
        # Aggregate tags per movie
        # Apply TF-IDF
        self.tag_features = tfidf.fit_transform(aggregated_tags)
        
    def build_content_features(self):
        """Combine genre and tag features"""
        # Horizontal stack or weighted combination
        pass
        
    def compute_similarity(self):
        """Calculate content similarity matrix"""
        self.content_similarity = cosine_similarity(content_features)
        
    def recommend(self, user_id, user_profile, k=10):
        """Recommend similar movies to user's profile"""
        # Create user content profile from history
        # Find movies with highest content similarity
        pass
```

**Feature Engineering**:
- **Genre Features**: 19-dimensional binary vector (one per genre)
- **Tag Features**: 100-dimensional TF-IDF vector
- **Combined Features**: Concatenated or weighted combination

### 4.4 Hybrid Recommendation Component

**Fusion Strategy**: Linear combination of collaborative and content-based scores

```python
class HybridRecommender:
    def __init__(self, alpha=0.7, beta=0.3):
        self.alpha = alpha  # Collaborative weight
        self.beta = beta    # Content-based weight
        self.cf_model = None
        self.content_model = None
        
    def recommend(self, user_id, k=10):
        """Generate hybrid recommendations"""
        # Get CF scores
        cf_scores = self.cf_model.recommend(user_id, k=50)
        
        # Get content scores
        content_scores = self.content_model.recommend(user_id, k=50)
        
        # Normalize scores to [0, 1]
        cf_normalized = normalize_scores(cf_scores)
        content_normalized = normalize_scores(content_scores)
        
        # Combine scores
        hybrid_scores = {}
        for movie_id in set(cf_scores.keys()) | set(content_scores.keys()):
            hybrid_scores[movie_id] = (
                self.alpha * cf_normalized.get(movie_id, 0) +
                self.beta * content_normalized.get(movie_id, 0)
            )
        
        # Apply diversity re-ranking
        final_recs = self.diversify(hybrid_scores, k)
        
        return final_recs
```

**Weight Optimization**:
- Grid search over α ∈ [0, 1], β = 1 - α
- Optimize for NDCG@10 on validation set
- Cross-validation to prevent overfitting

### 4.5 Explainability Component

**Explanation Generation Strategy**:

```python
class ExplainabilityEngine:
    def __init__(self):
        self.movies_df = None
        self.user_history = None
        
    def explain_recommendation(self, user_id, movie_id):
        """Generate natural language explanation"""
        # 1. Find user's highly-rated movies
        user_top_movies = self.get_user_top_movies(user_id, min_rating=4.0)
        
        # 2. Find most similar movie from user's history
        most_similar_movie = self.find_most_similar(
            movie_id, 
            user_top_movies
        )
        
        # 3. Identify shared attributes
        shared_genres = self.get_shared_genres(movie_id, most_similar_movie)
        shared_tags = self.get_shared_tags(movie_id, most_similar_movie)
        
        # 4. Generate explanation text
        if shared_genres:
            return f"Because you liked '{most_similar_movie}', " \
                   f"which shares genres {shared_genres}"
        elif shared_tags:
            return f"Because you liked '{most_similar_movie}', " \
                   f"which shares tags {shared_tags}"
        else:
            return "Recommended by similar users"
    
    def find_most_similar(self, movie_id, candidate_movies):
        """Find movie with highest content similarity"""
        similarities = []
        for candidate in candidate_movies:
            sim = self.compute_similarity(movie_id, candidate)
            similarities.append((candidate, sim))
        return max(similarities, key=lambda x: x[1])[0]
```

**Explanation Types**:
1. **Genre-based**: "Shares genres ['Action', 'Sci-Fi']"
2. **Tag-based**: "Shares tags ['mind-bending', 'plot-twist']"
3. **Collaborative**: "Recommended by similar users who also liked X"
4. **Popularity**: "Highly rated by users with similar tastes"

### 4.6 Evaluation Component

**Metrics Implementation**:

```python
class EvaluationMetrics:
    def precision_at_k(self, relevant, recommended, k):
        """Precision@K = |relevant ∩ recommended| / k"""
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        hits = len([m for m in recommended_k if m in relevant_set])
        return hits / k
    
    def recall_at_k(self, relevant, recommended, k):
        """Recall@K = |relevant ∩ recommended| / |relevant|"""
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        hits = len([m for m in recommended_k if m in relevant_set])
        return hits / len(relevant_set) if relevant_set else 0
    
    def ndcg_at_k(self, relevant, recommended, k):
        """Normalized Discounted Cumulative Gain"""
        dcg = sum([
            (1 if recommended[i] in relevant else 0) / np.log2(i + 2)
            for i in range(min(k, len(recommended)))
        ])
        
        idcg = sum([
            1 / np.log2(i + 2)
            for i in range(min(k, len(relevant)))
        ])
        
        return dcg / idcg if idcg > 0 else 0
    
    def intra_list_diversity(self, recommended_movies):
        """Average pairwise dissimilarity"""
        n = len(recommended_movies)
        if n <= 1:
            return 0
        
        total_dissimilarity = 0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # 1 - similarity between movies i and j
                dissim = 1 - self.movie_similarity[
                    recommended_movies[i], 
                    recommended_movies[j]
                ]
                total_dissimilarity += dissim
                count += 1
        
        return total_dissimilarity / count
    
    def catalog_coverage(self, all_recommendations):
        """Fraction of catalog that appears in any recommendation"""
        unique_recommended = set()
        for user_recs in all_recommendations:
            unique_recommended.update(user_recs)
        
        return len(unique_recommended) / self.total_movies
    
    def novelty_score(self, recommended_movies):
        """Average -log2(popularity) of recommendations"""
        novelty_scores = []
        for movie_id in recommended_movies:
            popularity = self.movie_popularity.get(movie_id, 1)
            novelty = -np.log2(popularity / self.total_ratings)
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)
```

---

## 5. Algorithm Implementation

### 5.1 Collaborative Filtering Algorithm

**User-User CF**:

```
Algorithm: User-User Collaborative Filtering
Input: user_id, k (number of recommendations)
Output: List of (movie_id, score) tuples

1. Retrieve user_vector from user-item matrix
2. Compute similarities with all other users
3. Select top-N most similar users (neighbors)
4. For each unrated movie:
   a. Find neighbors who rated it
   b. Compute weighted average: 
      score = Σ(similarity[i] × rating[i]) / Σ(similarity[i])
5. Sort movies by predicted score
6. Return top-k movies
```

**Time Complexity**: O(|U| + |I|) where U = users, I = items  
**Space Complexity**: O(|U|²) for similarity matrix

### 5.2 Content-Based Filtering Algorithm

```
Algorithm: Content-Based Filtering
Input: user_id, k (number of recommendations)
Output: List of (movie_id, score) tuples

1. Build user content profile:
   user_profile = Σ(rating[i] × content_vector[i]) / Σ(rating[i])
   for all movies i rated by user

2. For each candidate movie j:
   score[j] = cosine_similarity(user_profile, content_vector[j])

3. Filter out already-rated movies
4. Sort by score descending
5. Return top-k movies
```

**Time Complexity**: O(|I| × d) where d = feature dimension  
**Space Complexity**: O(|I| × d) for content vectors

### 5.3 Hybrid Recommendation Algorithm

```
Algorithm: Hybrid Recommendation with Diversity
Input: user_id, k, α (collaborative weight), β (content weight)
Output: Diversified list of (movie_id, score) tuples

1. Generate CF recommendations:
   cf_recs = collaborative_filtering(user_id, k=50)

2. Generate content recommendations:
   content_recs = content_based_filtering(user_id, k=50)

3. Normalize scores to [0, 1]:
   cf_norm = minmax_normalize(cf_recs)
   content_norm = minmax_normalize(content_recs)

4. Compute hybrid scores:
   for movie_id in (cf_recs ∪ content_recs):
       hybrid_score[movie_id] = α × cf_norm[movie_id] + 
                                 β × content_norm[movie_id]

5. Diversity re-ranking:
   selected = []
   candidates = sort(hybrid_score, descending)
   
   while len(selected) < k:
       best_score = -∞
       best_movie = None
       
       for movie in candidates:
           if movie in selected:
               continue
           
           # Balance relevance and diversity
           diversity_bonus = min_similarity(movie, selected)
           adjusted_score = hybrid_score[movie] + λ × diversity_bonus
           
           if adjusted_score > best_score:
               best_score = adjusted_score
               best_movie = movie
       
       selected.append(best_movie)

6. Return selected movies with scores
```

**Parameters**:
- α = 0.7 (collaborative weight)
- β = 0.3 (content weight)
- λ = 0.2 (diversity weight)

---

## 6. Scalability Considerations

### 6.1 Current Limitations

- **Memory**: Full similarity matrices for 9,742 movies = ~95M entries
- **Computation**: O(n²) similarity calculations
- **Real-time**: No online learning or incremental updates

### 6.2 Optimization Strategies

#### A. Sparse Matrix Operations
```python
from scipy.sparse import csr_matrix

# Use sparse matrices for user-item interactions
user_item_sparse = csr_matrix(user_item_matrix)

# Only store non-zero similarities
from scipy.sparse import lil_matrix
similarity_sparse = lil_matrix((n_movies, n_movies))
```

#### B. Approximate Nearest Neighbors
```python
from annoy import AnnoyIndex

# Build ANN index for fast similarity search
ann_index = AnnoyIndex(n_features, metric='angular')
for i, vector in enumerate(movie_vectors):
    ann_index.add_item(i, vector)
ann_index.build(n_trees=10)

# Fast k-NN queries
similar_movies = ann_index.get_nns_by_item(movie_id, k=20)
```

#### C. Dimensionality Reduction
```python
from sklearn.decomposition import TruncatedSVD

# Reduce feature dimensions
svd = TruncatedSVD(n_components=50)
reduced_features = svd.fit_transform(content_features)
```

#### D. Caching Strategy
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_recommendations(user_id, k=10):
    """Cache recent recommendations"""
    return hybrid_recommend(user_id, k)
```

### 6.3 Scalability Roadmap

| Scale | Users | Movies | Strategy |
|-------|-------|--------|----------|
| Small | <1K | <10K | In-memory, full matrices |
| Medium | 1K-100K | 10K-100K | Sparse matrices, ANN |
| Large | 100K-1M | 100K-1M | Distributed computing, sampling |
| Very Large | >1M | >1M | Sharding, online learning, GPU |

**Recommended Technologies**:
- **Apache Spark**: Distributed matrix factorization (ALS)
- **FAISS**: Fast approximate nearest neighbor search
- **Redis**: Caching layer for recommendations
- **PostgreSQL**: Efficient storage with indexing
- **TensorFlow**: Deep learning-based recommendations

---

## 7. Future Enhancements

### 7.1 Short-Term Improvements

1. **Cold Start Handling**
   - New user: Use demographic features or popular items
   - New item: Use content features until sufficient ratings

2. **Temporal Dynamics**
   - Time-decay for older ratings
   - Trend-aware recommendations
   - Session-based recommendations

3. **Context-Aware Recommendations**
   - Time of day (weekend vs. weekday)
   - Device (mobile vs. desktop)
   - Mood detection from recent activity

### 7.2 Medium-Term Enhancements

1. **Deep Learning Models**
   - Neural Collaborative Filtering (NCF)
   - Variational Autoencoders (VAE)
   - Graph Neural Networks (GNN)

2. **Multi-Armed Bandit**
   - Exploration-exploitation trade-off
   - A/B testing framework
   - Online learning

3. **Advanced Explainability**
   - Counterfactual explanations
   - Visual explanations (poster similarity)
   - Multi-factor explanations

### 7.3 Long-Term Vision

1. **Conversational Recommendations**
   - Natural language queries
   - Iterative refinement
   - Preference elicitation dialogue

2. **Multi-Stakeholder Optimization**
   - User satisfaction
   - Content provider diversity
   - Platform revenue

3. **Federated Learning**
   - Privacy-preserving recommendations
   - Decentralized model training
   - User data ownership

---

## 8. Technical Decisions & Rationale

### 8.1 Why Hybrid Approach?

**Problem**: No single algorithm excels at all aspects

| Algorithm | Strengths | Weaknesses |
|-----------|-----------|------------|
| Collaborative | Discovers patterns, diverse | Cold start, popularity bias |
| Content-Based | No cold start, serendipity | Filter bubbles, limited diversity |
| Hybrid | Best of both worlds | Complexity, parameter tuning |

**Decision**: Weighted hybrid with α=0.7 for collaborative, β=0.3 for content

### 8.2 Why Cosine Similarity?

**Alternatives**: Euclidean distance, Pearson correlation, Jaccard

**Chosen**: Cosine similarity
- **Reason**: Scale-invariant, works well with sparse data
- **Trade-off**: Doesn't account for rating scale differences

### 8.3 Why Leave-Last-Out Split?

**Alternatives**: Random split, k-fold cross-validation

**Chosen**: Temporal leave-last-out
- **Reason**: Simulates real-world scenario (predict future from past)
- **Trade-off**: Less data for training, harder evaluation

---

## 9. Deployment Architecture (Future)

```
┌────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
└───────────────────────┬────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│  API       │  │  API       │  │  API       │
│  Server 1  │  │  Server 2  │  │  Server 3  │
└─────┬──────┘  └─────┬──────┘  └─────┬──────┘
      │               │               │
      └───────────────┼───────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌────────────┐  ┌──────────┐  ┌──────────┐
│  Redis     │  │ Database │  │  Model   │
│  Cache     │  │ (Postgres│  │  Store   │
└────────────┘  └──────────┘  └──────────┘
```

---

## 10. References & Further Reading

1. **Matrix Factorization**: Koren, Y. (2009). Matrix factorization techniques for recommender systems.

2. **Hybrid Systems**: Burke, R. (2002). Hybrid recommender systems: Survey and experiments.

3. **Explainability**: Zhang, Y., & Chen, X. (2020). Explainable recommendation: A survey and new perspectives.

4. **Diversity**: Adomavicius, G., & Kwon, Y. (2012). Improving aggregate recommendation diversity using ranking-based techniques.

5. **Evaluation**: Herlocker, J. L., et al. (2004). Evaluating collaborative filtering recommender systems.

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Maintained By**: ReelSense Development Team
