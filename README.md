# Euro2024 Sentiment Analysis

## Analysis of Public Perception of Women's Participation in Football during Euro 2024

![EURO-2024](https://github.com/Munchkinland/Euro2024-Sentimel-Analysis/assets/92251234/ae209ede-8a66-4990-950b-7d105c31cd08)

### Introduction

The importance of equal opportunities for women in sports cannot be overstated. Historically, sports have been a male-dominated arena, with women often facing significant barriers to participation, recognition, and equal treatment. However, the landscape is gradually changing, and events like the Euro 2024 are prime examples of platforms where women athletes are increasingly showcasing their talent and dedication.

### Why Analyzing Public Perception of Women's Participation in Football during Euro 2024 is Crucial

- **Understanding Public Sentiment**: Gauging how the public perceives women's participation helps stakeholders understand the level of support or opposition. This can influence policies, sponsorship, and media coverage.
- **Identifying Areas for Improvement**: Sentiment analysis can highlight specific areas where women's participation is either praised or criticized, allowing organizations to address these issues effectively.
- **Promoting Equality**: By continuously monitoring and analyzing public opinion, we can promote equality in sports, ensuring that women receive the recognition and opportunities they deserve.
- **Supporting Decision Making**: Organizations, advertisers, and policy-makers can use these insights to make informed decisions that support and promote women's sports.

## Purpose of the Analysis

The purpose of this analysis is to understand how the participation of women in football during Euro 2024 is perceived. Through sentiment analysis of posts and comments on Reddit, we aim to identify positive, negative, and neutral opinions on this topic. This information can be useful for sports organizations, journalists, and analysts who wish to understand public perception and make informed decisions.

## Project Setup

### APIs and Libraries Used

- **PRAW (Python Reddit API Wrapper)**: To access posts and comments on Reddit.
- **Transformers by Hugging Face**: To use pre-trained sentiment analysis models.
- **NLTK (Natural Language Toolkit)**: For sentence tokenization.
- **Plotly**: For data visualization.

### Model Used (Pipelines)

We used the `cardiffnlp/twitter-roberta-base-sentiment` sentiment analysis model provided by Hugging Face. This model is optimized for analyzing sentiments in short texts, such as social media posts and comments.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/tu-usuario/nombre-del-repositorio.git
    cd nombre-del-repositorio
    ```

2. **Create and activate a virtual environment** (optional but recommended):
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Linux/Mac
    myenv\Scripts\activate     # On Windows
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure your Reddit API credentials**:
    Create a file named `config.ini` in the root directory of the project with the following content:
    ```ini
    [reddit]
    client_id = YOUR_CLIENT_ID
    client_secret = YOUR_CLIENT_SECRET
    user_agent = YOUR_USER_AGENT
    ```

## Usage

1. **Run the sentiment analysis script**:
    ```bash
    python analisis_sentimientos.py
    ```

2. **Visualize the results**:
    The script will output various visualizations using Plotly, including bar charts, scatter plots, pie charts, and box plots.

## Explanation of the Code

### Import Libraries and Configure Models

We import the necessary libraries, including PRAW for accessing Reddit, NLTK for text processing, Hugging Face's Transformers for sentiment analysis, and Plotly for data visualization. We also configure the sentiment analysis model.

### Functions for Text Processing and Sentiment Analysis

- `split_text_into_chunks(text, tokenizer, max_length=512)`: Splits text into chunks that fit within the model's maximum token length.
- `analyze_text_chunks(text)`: Analyzes each chunk of text for sentiment.
- `aggregate_sentiments(sentiments)`: Aggregates the sentiment scores of the chunks into a single score.

### Fetch Data from Reddit

- `fetch_reddit_posts_and_comments()`: Fetches posts and comments from the `euro2024` subreddit that mention "women".

### Processing and Sentiment Analysis

We fetch the texts, process them to analyze sentiments, and aggregate the results.

### Calculate Loss Rate and Display Results

We calculate the total number of texts processed, the number of errors, and the loss rate. The results are printed to the console.

### Data Visualization

We use Plotly to create various visualizations to represent the sentiment analysis results.

- **Bar Chart**: Shows the distribution of sentiments.
- **Scatter Plot**: Displays sentiment scores for each text.
- **Pie Chart**: Illustrates the proportion of each sentiment type.
- **Box Plot**: Shows the distribution of sentiment scores within each sentiment category.

## Conclusions

Through this analysis, we can draw several key conclusions about the public perception of women's participation in football during Euro 2024:

- **Distribution of Sentiments**: Most of the analyzed texts exhibit a neutral sentiment, followed by positive and then negative sentiments.
- **Proportion of Sentiments**: The pie chart clearly shows the proportion of each sentiment type, indicating that the perception is mostly neutral or positive.
- **Sentiment Scores**: The box plot shows the distribution of sentiment scores, indicating variability within each sentiment category.

This analysis provides a clear view of how the participation of women in football during Euro 2024 is perceived on the Reddit platform, helping guide future strategies and communications in the sports and social spheres.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## Acknowledgements

Thanks to [Hugging Face](https://huggingface.co/) for providing the sentiment analysis models and to [Plotly](https://plotly.com/) for the visualization tools.


