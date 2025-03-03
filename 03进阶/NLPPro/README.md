# NLPPro: Financial News Sentiment Analysis and Stock Trend Prediction

## Project Overview
This project aims to analyze financial news sentiment and predict stock trends using advanced machine learning techniques, specifically LSTM and Transformers. By leveraging natural language processing (NLP) and time series analysis, we aim to provide insights into stock market movements based on news sentiment.

## Project Structure
```
NLPPro
├── data
│   ├── raw                # Folder for raw financial news data
│   ├── processed          # Folder for processed financial news data
├── models
│   ├── lstm_model.py      # LSTM model definition and training
│   └── transformers_model.py # Transformers model definition and training
├── notebooks
│   └── exploratory_data_analysis.ipynb # Jupyter Notebook for exploratory data analysis
├── src
│   ├── data_preprocessing.py # Data preprocessing functions
│   ├── sentiment_analysis.py  # Sentiment analysis functions
│   ├── stock_prediction.py     # Stock trend prediction functions
│   └── utils.py               # Utility functions
├── requirements.txt           # Required Python libraries
└── README.md                  # Project documentation
```

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd NLPPro
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. **Data Preprocessing**: Use `src/data_preprocessing.py` to load and clean the financial news data.
2. **Sentiment Analysis**: Run `src/sentiment_analysis.py` to analyze the sentiment of the news articles and extract features.
3. **Stock Prediction**: Execute `src/stock_prediction.py` to train and evaluate the stock trend prediction models using the processed data.
4. **Exploratory Data Analysis**: Open `notebooks/exploratory_data_analysis.ipynb` for visualizations and preliminary statistical analysis.

## Results
The results of the sentiment analysis and stock trend predictions will be available in the output of the respective scripts. Visualizations can be found in the Jupyter Notebook.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.