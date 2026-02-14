# SentimentStocks

An exploratory machine learning project that uses natural language processing and financial news sentiment to predict stock price movements.

## Overview

This project investigates whether financial news events, as captured by the GDELT (Global Database of Events, Language, and Tone) dataset, contain predictive signals for stock price direction. By combining news event data with sophisticated NLP embeddings and a custom multimodal neural network architecture, the system attempts to extract actionable insights from high-noise financial inputs.

## Motivation

Financial markets are notoriously difficult to predict due to their chaotic nature and sensitivity to countless variables. This project explores whether large-scale event data from global news sources can provide an edge in predicting short-term price movements. Rather than relying solely on technical indicators or traditional sentiment analysis, this approach processes structured event representations including actors, event types, geographic information, and media attention metrics.

## Architecture

### Data Pipeline

The data processing pipeline ([dataSetup2.ipynb](dataSetup2.ipynb)) handles the ingestion, transformation, and integration of two primary data sources:

- **GDELT Event Data**: Global news events with structured metadata including actors, event codes (CAMEO taxonomy), geographic coordinates, and mention counts
- **Stock Price Data**: Historical daily stock prices used to generate binary labels (price increase vs. decrease)

Key pipeline features:
- Processes 100GB+ of time-series financial and news data
- Custom embedding lookups for event codes (512-dimensional) and actors (300-dimensional GloVe embeddings)
- Efficient chunking strategy to handle memory constraints
- HDF5 and compressed pickle formats for optimized storage and retrieval
- Data validation and error handling for missing or malformed entries

### Model Architecture

The prediction model ([stockAnalysis.ipynb](stockAnalysis.ipynb)) implements a multimodal neural network in TensorFlow that processes four distinct input streams:

1. **EventCode Embeddings** (512-dim): Encoded representation of event types using CAMEO taxonomy
2. **Actor1 Embeddings** (300-dim): GloVe embeddings for primary actors in news events
3. **Actor2 Embeddings** (300-dim): GloVe embeddings for secondary actors in news events
4. **Singular Features** (6-dim): Numerical features including:
   - Number of mentions
   - Number of sources
   - Geographic coordinates (latitude/longitude) for both actors

The model architecture:
- Parallel processing branches for each input type
- Dense layers with ReLU activation for feature extraction
- Concatenation layer to merge multimodal features
- Dropout regularization (50%) to prevent overfitting
- Sigmoid output for binary classification (price up/down)

## Technical Implementation

### Technologies

- **TensorFlow**: Deep learning framework for model architecture and training
- **NumPy & Pandas**: Data manipulation and numerical processing
- **GloVe**: Pre-trained word embeddings (840B token, 300-dimensional)
- **HDF5**: Hierarchical data format for efficient storage of large datasets
- **GDELT**: Global Database of Events, Language, and Tone for news data
- **Hadoop**: Distributed processing for initial data ingestion and cleaning

### Data Processing Workflow

1. **Ingestion**: Load raw GDELT event data and stock price history
2. **Temporal Alignment**: Match news events to corresponding trading dates
3. **Embedding Lookup**: Map actors and event codes to pre-computed embeddings
4. **Label Generation**: Create binary labels based on next-day price movement
5. **Serialization**: Store processed data in optimized formats (HDF5, compressed pickle)
6. **Chunking**: Split data into manageable batches for training

### Model Training

- Custom training loop for fine-grained control over batch processing
- Train-test split with temporal awareness to prevent data leakage
- Adam optimizer with tuned learning rate (0.01)
- Binary cross-entropy loss function
- Accuracy tracking across training and validation sets

## Results

The model achieved **55%+ prediction accuracy** on held-out test data, demonstrating that financial news events contain statistically significant predictive signals above random chance (50%). While this represents a modest edge, it validates the hypothesis that structured event data can inform price prediction models.

Key observations:
- News event volume and source diversity showed correlation with price volatility
- Geographic proximity of events to market centers influenced signal strength
- Certain actor types (government entities, corporations) provided stronger signals
- The multimodal architecture outperformed single-input baselines

## Challenges & Learnings

### Data Challenges
- **Scale**: Managing 100GB+ datasets required careful memory optimization and chunking strategies
- **Noise**: Financial news contains substantial noise; distinguishing signal from randomness proved difficult
- **Temporal Dependencies**: News events have complex temporal relationships that simple models struggle to capture
- **Missing Data**: Incomplete GDELT records required robust error handling and validation

### Technical Challenges
- **Memory Constraints**: Initial approaches caused memory explosions; solved through HDF5 and incremental processing
- **Embedding Alignment**: Ensuring consistent embedding lookups across actors and event codes
- **Class Imbalance**: Slightly imbalanced distribution of price increases vs. decreases
- **Overfitting**: High-dimensional inputs required aggressive regularization

## Limitations

- **Short-term Prediction Only**: Model predicts next-day movements, not longer-term trends
- **Single Stock**: Analysis focused on a single equity; generalization unclear
- **No Transaction Costs**: Real-world trading would incur costs that erode slim margins
- **Data Delay**: GDELT data availability lags, limiting real-time applicability
- **Market Regime Changes**: Model trained on historical data may not generalize to different market conditions

## Future Directions

- Incorporate additional data sources (social media, earnings reports, technical indicators)
- Experiment with sequence models (LSTMs, Transformers) to capture temporal dependencies
- Multi-stock prediction to identify relative opportunities
- Attention mechanisms to identify which news events drive predictions
- Integration with portfolio optimization frameworks

## Project Structure

```
Stock-Analysis/
├── dataSetup2.ipynb          # Data processing and pipeline
├── stockAnalysis.ipynb        # Model architecture and training
├── .gitignore                 # Excludes large data files and embeddings
└── README.md                  # This file
```

## Data Files (Not Included)

Due to size constraints, the following files are excluded from the repository:
- `gdelt_data.csv` / `gdelt_data_cleaned.csv` - GDELT event data
- `stock_data.csv` - Historical stock prices
- `glove.840B.300d.txt` - GloVe word embeddings (4GB+)
- `cameo_embeddings.json` - Event code embeddings
- `actor_embeddings_cleaned.json` - Actor embeddings
- Processed data files (`.h5`, `.pkl.gz`)

## Usage

**Note**: This is an exploratory research project. The code is provided as-is for educational purposes and is not intended for production use or real trading decisions.

### Prerequisites
```bash
pip install tensorflow pandas numpy h5py joblib tqdm scikit-learn matplotlib
```

### Running the Pipeline
1. Obtain GDELT data and stock price data (not included)
2. Generate or obtain GloVe embeddings and CAMEO code mappings
3. Run [dataSetup2.ipynb](dataSetup2.ipynb) to process and prepare data
4. Run [stockAnalysis.ipynb](stockAnalysis.ipynb) to train and evaluate the model

## Disclaimer

This project is for educational and research purposes only. It should not be interpreted as financial advice. Financial markets are complex and unpredictable; past performance does not guarantee future results. Any trading decisions based on similar models carry substantial risk of financial loss.

## Acknowledgments

- GDELT Project for providing open access to global event data
- GloVe team at Stanford for pre-trained embeddings
- TensorFlow and scikit-learn communities for robust ML tools

---

*An exploratory project in machine learning and data science investigating the intersection of natural language processing and financial market prediction.*