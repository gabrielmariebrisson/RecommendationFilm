# 🎬 Ciné-Reco: Deep Learning Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Layered-blueviolet.svg)](https://en.wikipedia.org/wiki/Multitier_architecture)
[![CI](https://github.com/USERNAME/RecommendationFilm/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/RecommendationFilm/actions/workflows/ci.yml)

> **Production-ready movie recommendation system** using a Siamese Neural Network architecture, deployed as a scalable web application with async API integration and comprehensive test coverage.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Cold Start Problem](#cold-start-problem)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Technical Stack](#technical-stack)
- [Testing](#testing)
- [Production Considerations](#production-considerations)
- [Advanced Usage](#advanced-usage)

## 🎯 Overview

Ciné-Reco is a production-grade recommendation system that leverages deep learning to provide personalized movie recommendations. The system processes **2.8M+ ratings** from **8,493 users** across **20,763 movies** using a Siamese Neural Network architecture, achieving a **RMSE of 0.35** on the MovieLens dataset.

### Key Highlights

- **Deep Learning Model**: Siamese Neural Network with dual-tower architecture (User Tower / Item Tower)
- **Production Architecture**: Layered architecture with separation of concerns (Core/Service/Presentation)
- **Async Performance**: Parallel API calls using `aiohttp` for 20x faster metadata retrieval
- **Scalable Design**: Modular, testable, and maintainable codebase following PEP 8 standards
- **Internationalization**: Multi-language support with translation caching
- **MLOps Ready**: Model Registry with versioning, health checks, and structured logging
- **Performance Optimized**: Broadcasting-based memory optimization (80-90% reduction)
- **Production Monitoring**: Structured JSON logging with trace IDs and metrics tracking

## ✨ Key Features

- 🧠 **Deep Learning Recommendations**: Siamese Neural Network with custom similarity functions
- ⚡ **Async API Integration**: Parallel metadata fetching for instant poster display
- 🌍 **Multi-language Support**: 10 languages with intelligent caching
- 🏗️ **Layered Architecture**: Clean separation between business logic, services, and UI
- 🧪 **Comprehensive Testing**: 100% test coverage for critical paths with mocked dependencies
- 📊 **Real-time Predictions**: Sub-second inference time for personalized recommendations
- 🎨 **Modern UI**: Streamlit-based interface with responsive design
- 📦 **Model Registry**: Versioned model management with metadata tracking (accuracy, RMSE, commit hash)
- 🏥 **Health Checks**: Automatic inference validation on startup (Inference Ready)
- 📈 **Observability**: Structured JSON logging with trace IDs, metrics, and error tracking
- ⚡ **Performance Optimized**: Broadcasting-based memory optimization (O(1) vs O(N))
- 🧪 **Performance Benchmarks**: Scalability testing for 1M+ requests/day capacity

## 🏗️ Architecture

### Siamese Neural Network Design

The system employs a **dual-tower Siamese architecture** that learns joint embeddings for users and items in a shared latent space. This design choice offers several advantages over traditional collaborative filtering:

#### Architecture Components

```
┌─────────────────┐         ┌─────────────────┐
│   User Tower    │         │   Item Tower    │
│                 │         │                 │
│  Dense(256)     │         │  Dense(256)     │
│  BatchNorm      │         │  BatchNorm      │
│  GELU           │         │  GELU           │
│  Dropout(0.3)   │         │  Dropout(0.3)   │
│                 │         │                 │
│  Dense(128)     │         │  Dense(128)     │
│  BatchNorm      │         │  BatchNorm      │
│  GELU           │         │  GELU           │
│  Dropout(0.3)   │         │  Dropout(0.3)   │
│                 │         │                 │
│  Dense(64)      │         │  Dense(64)      │
│  L2 Normalize   │         │  L2 Normalize  │
└────────┬────────┘         └────────┬────────┘
         │                           │
         └───────────┬───────────────┘
                     │
         ┌───────────▼───────────┐
         │   Fusion Layer        │
         │                       │
         │  [diff_abs, prod_mul] │
         │                       │
         │  Dense(32)            │
         │  Dense(1, sigmoid)    │
         └───────────────────────┘
```

#### User Tower Input Features

The user embedding is constructed from:
- **Behavioral features**: Number of ratings, average rating
- **Genre preferences**: Mean rating per genre (computed from user's rating history)
- **Feature vector**: `[num_ratings, avg_rating, 0] + [pref_genre_1, ..., pref_genre_N]`

#### Item Tower Input Features

The item embedding encodes:
- **Content features**: Genre distribution, popularity metrics
- **Interaction patterns**: Historical rating patterns
- **Normalized features**: Pre-processed through `StandardScaler`

#### Custom Similarity Functions

The model uses two complementary similarity measures:

1. **Absolute Difference** (`diff_abs`): Captures dissimilarity between user and item embeddings
   ```python
   diff_abs = |user_embedding - item_embedding|
   ```

2. **Element-wise Product** (`prod_mul`): Models bilinear similarity and fine-grained interactions
   ```python
   prod_mul = user_embedding ⊙ item_embedding
   ```

These are concatenated and passed through dense layers to produce a compatibility score.

#### Why Siamese Architecture?

1. **Scalability**: Embeddings can be pre-computed and stored, enabling fast similarity search
2. **Generalization**: Learns transferable representations that work for new users/items
3. **Interpretability**: Embedding space can be visualized and analyzed
4. **Efficiency**: Single forward pass per user-item pair (O(1) complexity)

### Technical Implementation Details

- **Activation**: GELU (Gaussian Error Linear Unit) for smoother gradients
- **Regularization**: L2 regularization (1e-6) + Dropout (30%) to prevent overfitting
- **Normalization**: Batch Normalization + L2 normalization for stable training
- **Optimization**: Adam optimizer with learning rate scheduling
- **Loss Function**: Mean Squared Error (MSE) for regression task

## 🚀 Cold Start Problem

### Challenge

New users have no rating history, making it impossible to generate personalized recommendations using traditional collaborative filtering.

### Solution: Genre-Based Preference Inference

The system addresses the cold start problem through **intelligent genre preference estimation**:

```python
# For each genre, compute mean rating from user's history
genre_ratings = defaultdict(list)
for movie_id, rating in user_ratings.items():
    genres = movie_dict[movie_id]['genres']
    for genre in genres.split('|'):
        genre_ratings[genre].append(rating)

# Build preference vector: use genre-specific mean, fallback to global mean
user_prefs = {
    f'pref_{genre}': np.mean(genre_ratings.get(genre, [avg_rating]))
    for genre in unique_genres
}
```

#### How It Works

1. **Genre Extraction**: For each rated movie, extract all associated genres
2. **Genre-Level Aggregation**: Compute mean rating per genre from user's history
3. **Fallback Strategy**: For genres not yet rated, use the user's global average rating
4. **Feature Construction**: Build a preference vector with one feature per genre

#### Benefits

- **Immediate Personalization**: Works with as few as 3 ratings
- **Progressive Refinement**: Recommendations improve as user provides more ratings
- **Robust to Sparse Data**: Handles users with limited rating history gracefully
- **Content-Aware**: Leverages movie metadata (genres) for better cold start performance

#### Example

For a new user who rated:
- "The Matrix" (Action, Sci-Fi): 5.0
- "Inception" (Action, Sci-Fi, Thriller): 4.5
- "Titanic" (Drama, Romance): 3.0

The system infers:
- `pref_Action = 4.75` (mean of 5.0, 4.5)
- `pref_Sci-Fi = 4.75` (mean of 5.0, 4.5)
- `pref_Thriller = 4.5` (single rating)
- `pref_Drama = 3.0` (single rating)
- `pref_Romance = 3.0` (single rating)
- `pref_Other = 4.17` (global average for unseen genres)

This enables immediate personalized recommendations even for new users.

## 📦 Installation & Usage

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

#### Option 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RecommendationFilm
   ```

2. **Set up environment variables**
   ```bash
   # Create a .env file
   echo "API_KEY_FILM=your_omdb_api_key" > .env
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

   Or with Docker directly:
   ```bash
   docker build -t cinereco-app .
   docker run -p 8501:8501 --env-file .env cinereco-app
   ```

   The application will be available at `http://localhost:8501`

#### Option 2: Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RecommendationFilm
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create a .env file
   echo "API_KEY_FILM=your_omdb_api_key" > .env
   ```

5. **Verify model files are present**
   ```
   templates/assets/film/
   ├── best_model.keras
   ├── scalerUser.pkl
   ├── scalerItem.pkl
   ├── scalerTarget.pkl
   ├── movie_dict.pkl
   ├── item_vecs_finder.pkl
   └── unique_genres.pkl
   ```

### Usage

#### Docker (Recommended)

**Run with Docker Compose:**
```bash
docker-compose up
```

**Or with Docker directly:**
```bash
docker build -t cinereco-app .
docker run -p 8501:8501 --env-file .env cinereco-app
```

#### Local

**Run the Streamlit application:**
```bash
streamlit run RecommandationFilmsWeb.py
```

The application will be available at `http://localhost:8501`

**Usage Flow:**
1. Rate at least 3 movies in the sidebar
2. View personalized recommendations in the main panel
3. Filter recommendations by genre
4. Explore detailed movie information (actors, plot, IMDb rating)

## 🐳 Docker Deployment

### Quick Start

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t cinereco-app .
docker run -p 8501:8501 --env-file .env cinereco-app
```

### Docker Compose

The `docker-compose.yml` file provides a complete setup:

```bash
# Start services
docker-compose up

# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Image Details

- **Base Image**: Python 3.11-slim (optimized size)
- **Multi-stage Build**: Reduces final image size
- **Non-root User**: Runs as `appuser` for security
- **Health Check**: Automatic health monitoring
- **Port**: 8501 (Streamlit default)

### Environment Variables

Create a `.env` file or pass environment variables:

```bash
# .env file
API_KEY_FILM=your_omdb_api_key
```

Or pass directly:
```bash
docker run -p 8501:8501 -e API_KEY_FILM=your_key cinereco-app
```

### Volumes

Model files are mounted as read-only volumes:
- `./templates/assets/film:/app/templates/assets/film:ro`

This allows updating models without rebuilding the image.

### Production Deployment

For production, consider:

1. **Use a reverse proxy** (nginx, Traefik)
2. **Add SSL/TLS** certificates
3. **Set resource limits** in docker-compose.yml
4. **Use Docker secrets** for sensitive data
5. **Enable logging** aggregation

Example production docker-compose:

```yaml
services:
  recommendation-app:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Development

**Run tests:**

**Run tests:**
```bash
# All tests
pytest tests/ -v

# Specific test suite
pytest tests/test_metadata.py -v
pytest tests/test_recommender.py -v

# With coverage
pytest tests/ --cov=core --cov=services --cov-report=html
```

**CI/CD:**
The project includes GitHub Actions workflows for automated testing and quality checks:
- **CI Workflow**: Runs on every push/PR (tests, linting, security scans)
- **Performance Benchmarks**: Weekly performance testing
- **Release Workflow**: Automated release notes generation

See `.github/workflows/` for details.

**Code quality:**
```bash
# Type checking (if mypy is installed)
mypy src/ RecommandationFilmsWeb.py

# Linting
flake8 src/ RecommandationFilmsWeb.py
```

## 📁 Project Structure

The project follows a **strict layered architecture** pattern, ensuring separation of concerns and maintainability:

```
RecommendationFilm/
├── RecommandationFilmsWeb.py  # Presentation layer (Streamlit UI)
├── requirements.txt           # Python dependencies
│
├── src/                      # Source code package
│   ├── __init__.py
│   ├── config.py            # Centralized configuration
│   │
│   ├── core/                 # Business logic layer
│   │   ├── __init__.py
│   │   ├── recommender.py   # MovieRecommender class
│   │   │                    # - Model loading & inference
│   │   │                    # - Recommendation generation
│   │   │                    # - User preference computation
│   │   │                    # - Health checks
│   │   ├── model_registry.py # Model Registry & versioning
│   │   │                     # - ModelVersionManager
│   │   │                     # - Metadata validation
│   │   │                     # - Version management
│   │   └── monitoring.py    # Observability layer
│   │                        # - StructuredLogger (JSON logs)
│   │                        # - RecommendationMetrics
│   │                        # - Trace ID generation
│   │
│   ├── services/             # Service layer
│   │   ├── __init__.py
│   │   └── metadata.py      # External API services
│   │                        # - MetadataService (OMDb API)
│   │                        # - TranslationService (i18n)
│   │                        # - Async batch operations
│   │
│   ├── scripts/              # Utility scripts
│   │   ├── register_model.py # Model registration CLI
│   │   └── README.md        # Scripts documentation
│   │
│   └── models/               # Model Registry storage
│
├── tests/                    # Test suite
│   ├── conftest.py          # Pytest configuration & mocks
│   ├── test_metadata.py     # API service tests
│   ├── test_recommender.py  # Core logic tests
│   └── performance/         # Performance benchmarks
│       ├── benchmark_scale.py # Scalability tests
│       └── generate_report.py # Benchmark reports
│
└── templates/                # Static assets
    └── assets/
        ├── film/            # Model files & scalers
        └── images/          # UI assets
```

### Layer Responsibilities

#### **Core Layer** (`src/core/`)
- **Purpose**: Business logic and domain models
- **Responsibilities**:
  - Model inference and prediction
  - Recommendation algorithm implementation
  - User preference computation
  - Data transformation and normalization
  - Model versioning and registry management
  - Health checks and observability
  - Structured logging and metrics
- **Dependencies**: TensorFlow, NumPy, Pandas
- **No dependencies on**: Streamlit, external APIs
- **Key Modules**:
  - `recommender.py`: Main recommendation engine with health checks
  - `model_registry.py`: Model versioning and metadata management
  - `monitoring.py`: Structured logging and metrics tracking

#### **Service Layer** (`src/services/`)
- **Purpose**: External integrations and cross-cutting concerns
- **Responsibilities**:
  - API communication (OMDb, translation services)
  - Async request handling
  - Caching strategies
  - Error handling for external services
- **Dependencies**: aiohttp, requests, deep-translator
- **No dependencies on**: Core business logic

#### **Presentation Layer** (`RecommandationFilmsWeb.py`)
- **Purpose**: User interface and interaction
- **Responsibilities**:
  - Streamlit UI components
  - User input handling
  - Result visualization
  - State management

#### **Streamlit Helpers** (`src/streamlit_helpers.py`)
- **Purpose**: Initialization and helper functions for Streamlit
- **Responsibilities**:
  - Service initialization (recommender, metadata, translation)
  - Streamlit cache management
  - UI utility functions
- **Dependencies**: Streamlit, Core layer, Service layer
- **Thin layer**: Delegates all business logic to Core

### Design Principles

- **Dependency Inversion**: High-level modules don't depend on low-level modules
- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Open for extension, closed for modification
- **Interface Segregation**: Clean interfaces between layers
- **Dependency Injection**: Services injected via constructors

## 📊 Performance Metrics

### Model Performance

- **RMSE**: 0.35 (Root Mean Square Error)
- **MSE**: 0.12 (Mean Square Error)
- **Training Time**: 35 minutes on MacBook M1 (CPU only)
- **Model Size**: 1.8 MB (lightweight deployment)

### System Performance

- **Inference Time**: ~234ms average for 20,000+ movie recommendations
- **API Latency**: ~500ms for 20 parallel metadata requests (vs ~10s sequential)
- **Memory Footprint**: < 500 MB (including model and data)
- **Scalability**: Handles 20,763 movies with sub-second response time
- **Throughput**: 50+ requests/second (validated with benchmarks)
- **Daily Capacity**: 1M+ requests/day (conservative estimate)
- **Memory Optimization**: 80-90% reduction via broadcasting (O(1) vs O(N))

### Dataset Statistics

- **Movies**: 20,763 (1874-2024)
- **Users**: 8,493 active users
- **Ratings**: 2,864,752 total ratings
- **Genres**: 20 unique genres
- **Rating Scale**: 0.5 to 5.0 stars

## 🛠️ Technical Stack

### Core Technologies

- **Deep Learning**: TensorFlow 2.20 / Keras 3.11
- **Web Framework**: Streamlit 1.50
- **Data Processing**: Pandas 2.3, NumPy 2.3
- **Async HTTP**: aiohttp 3.11
- **Testing**: pytest 8.4

### Key Libraries

- **Machine Learning**: scikit-learn (scalers), TensorFlow (neural networks)
- **API Integration**: aiohttp (async), requests (sync fallback)
- **Internationalization**: deep-translator
- **Configuration**: python-dotenv
- **Type Safety**: Full type hints (Python 3.10+)

### Code Quality

- **Type Hints**: 100% coverage with `typing` module
- **PEP 8 Compliance**: Strict adherence to Python style guide
- **Error Handling**: Specific exception types (no bare `except`)
- **Path Management**: `pathlib.Path` for cross-platform compatibility
- **Documentation**: Comprehensive docstrings and type annotations

## 🧪 Testing

### Test Coverage

- **Unit Tests**: Core recommendation logic
- **Integration Tests**: API service interactions
- **Mock Strategy**: All external dependencies mocked (no real API calls)
- **Test Count**: 20+ test cases covering:
  - API error handling (404, timeout, parsing errors)
  - Cold start scenarios
  - Data consistency validation
  - Edge cases (empty inputs, missing data)

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=core --cov=services --cov-report=html

# Run specific test file
pytest tests/test_metadata.py -v
```

### Test Architecture

- **Fixtures**: Reusable test data and mocks
- **Isolation**: Each test is independent
- **Speed**: All tests run in < 2 seconds (no I/O)
- **Reliability**: Deterministic results with mocked dependencies

## 🚀 Production Considerations

### Scalability

- **Horizontal Scaling**: Stateless design allows multiple instances
- **Caching**: Streamlit cache for model and API responses
- **Async Operations**: Non-blocking I/O for external APIs
- **Batch Processing**: Efficient vectorized operations

### Monitoring & Observability

- **Structured Logging**: JSON-formatted logs with trace IDs for correlation
- **Metrics Tracking**: Automatic recording of inference latency, throughput, cache hits
- **Error Tracking**: Full stacktraces with error types and context
- **Health Checks**: Automatic model validation on startup (Inference Ready)
- **Model Registry**: Versioned models with metadata (accuracy, RMSE, commit hash)
- **Performance Benchmarks**: Scalability testing tools for capacity planning

#### Logging Example

```json
{
  "timestamp": "2025-01-27T10:30:45.123456+00:00",
  "level": "INFO",
  "service": "movie-recommender",
  "message": "recommendation_metrics",
  "trace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "num_input_ratings": 5,
  "num_recommendations": 150,
  "inference_time_ms": 234.5,
  "cache_hit": false
}
```

### Model Management

- **Model Registry**: Versioned storage with automatic latest stable selection
- **Metadata Validation**: Automatic checks for model files and dependencies
- **Health Checks**: Dummy prediction on startup to verify model functionality
- **Git Integration**: Automatic commit hash tracking for model versions
- **Rollback Support**: Easy reversion to previous stable versions

#### Registering a Model

```bash
python scripts/register_model.py v1 \
    --rmse 0.85 \
    --accuracy 0.92 \
    --description "Production model v1"
```

### Performance Optimization

- **Broadcasting Optimization**: Memory-efficient tensor operations (O(1) vs O(N))
- **Single Scaler Transform**: Transform user vector once instead of N times
- **Vectorized Operations**: NumPy/TensorFlow for efficient batch processing
- **Benchmark Suite**: Automated scalability testing (100, 1k, 10k concurrent users)


## 🔧 Advanced Usage

### Model Registry

The system includes a Model Registry for versioned model management:

```python
from src.core.model_registry import ModelVersionManager
from src.config import MODEL_REGISTRY_PATH

# Initialize registry
registry = ModelVersionManager(registry_path=MODEL_REGISTRY_PATH)

# Load latest stable version
paths = registry.load_model_paths()
recommender = MovieRecommender(
    model_registry=registry,
    model_version=None  # Uses latest stable
)
```

### Health Checks

Automatic health checks validate model functionality:

```python
recommender = MovieRecommender()
recommender.initialize()  # Health check runs automatically

# Manual health check
is_healthy, error = recommender.health_check()
if not is_healthy:
    print(f"Model unhealthy: {error}")
```

### Observability

Structured logging with trace IDs:

```python
from src.core.monitoring import get_logger

logger = get_logger()
logger.info(
    "Processing recommendation",
    trace_id="abc123",
    user_id=123,
    num_ratings=5
)
```

### Performance Benchmarks

Run scalability tests:

```bash
cd tests/performance
python benchmark_scale.py

# Generate benchmark report
python generate_report.py
```

## 📝 License

MIT License - see LICENSE file for details

## 👤 Author

**Gabriel Marie-Brisson**

- Portfolio: [gabriel.mariebrisson.fr](https://gabriel.mariebrisson.fr)

---

**Built with ❤️ using TensorFlow, Streamlit, and modern Python practices**

**Production-Ready Features**: Model Registry, Health Checks, Observability, Performance Optimization

