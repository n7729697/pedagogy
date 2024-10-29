import spacy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import matplotlib.pyplot as plt
import numpy as np
import nltk
nltk.download('vader_lexicon')

# Load spacy model
nlp = spacy.load('en_core_web_sm')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Example text for processing
folder_path = '/path/to/your/folder'  # Change to your .txt files directory

# Define the aspects to analyze
aspects = ['engage', 'privacy', 'access', 'teaching', 'learning']

# List to store the results
results = []

# Process each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # Process only .txt files
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Process text using Spacy
        doc = nlp(text)
        
        # Dictionary to store sentiment scores for each aspect
        aspect_sentiment = {aspect: [] for aspect in aspects}
        
        # Extract sentences and perform sentiment analysis for each aspect
        for sent in doc.sents:
            for aspect in aspects:
                if aspect in sent.text.lower():
                    sentiment = sid.polarity_scores(sent.text)
                    # Store the compound sentiment score (overall sentiment)
                    aspect_sentiment[aspect].append(sentiment['compound'])
        
        # Compute overall sentiment for each aspect (mean of compound scores)
        overall_sentiment = {aspect: np.mean(scores) if scores else 0 for aspect, scores in aspect_sentiment.items()}
        
        # Add the result for this file to the list
        result = {'file': filename.replace('.txt', '')}
        result.update(overall_sentiment)
        results.append(result)

# Convert results to a DataFrame
df = pd.DataFrame(results)

# Save the results to a CSV file
output_csv = '/home/xuezhi/Desktop/aspect_sentiment_analysis_results.csv'
df.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")
