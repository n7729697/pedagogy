import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import pandas as pd
import os
import argparse
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
# Set the font to Times New Roman
#plt.rcParams['font.family'] = 'Times New Roman'
# Initialize lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
unwanted_words = {'ai', 'genai', 'generative'}

# Custom preprocessing function
def preprocess_text(text):
    # Step 1: Tokenization (Split text into words)
    tokens = word_tokenize(text.lower())
    
    # Step 2: Remove punctuation
    tokens = [word for word in tokens if word.isalpha()]
    
    # Step 3: Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    
    # Step 4: Lemmatization (Convert words to base form)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

# Function to calculate TF-IDF and save to individual CSV files
def process_file(file_path, output_dir, plot):
    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Preprocess the text
    preprocessed_text = ' '.join(preprocess_text(text))
   
    # Remove 'ai' from the preprocessed text for word counts
    filtered_text = [word for word in preprocess_text(text) if word not in unwanted_words]
    
    # Calculate word count frequencies (without "ai")
    word_counts = Counter(filtered_text)
    
    # print(preprocessed_text)
    # Apply TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_text])
    
    # Get terms and their TF-IDF values
    terms = tfidf_vectorizer.get_feature_names_out()
    sum_tfidf = np.array(tfidf_matrix.sum(axis=0)).flatten()
    term_tfidf_dict = dict(zip(terms, sum_tfidf))
    
    # print(sorted(term_tfidf_dict.items(), key=lambda item: item[1], reverse=True)[:10])
    # Combine word count frequencies and TF-IDF values into a DataFrame
    all_words = set(word_counts.keys()).union(set(term_tfidf_dict.keys()))  # Combine all terms from counts and TF-IDF
    
    word_data = {
        'Term': list(all_words),
        'Word Count': [word_counts.get(word, 0) for word in all_words],
        'TF-IDF': [term_tfidf_dict.get(word, 0) for word in all_words]
    }
    
    df = pd.DataFrame(word_data)
    
    # Save the DataFrame to a CSV file using the original file's name
    base_filename = os.path.basename(file_path).replace('.txt', '.csv')
    output_csv_path = os.path.join(output_dir, base_filename)
    df.to_csv(output_csv_path, index=False)
    print(f"TF-IDF frequencies saved to {output_csv_path}")
    
    # Generate a word cloud if --plot is specified
    if plot:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(term_tfidf_dict)
        output_image_path = os.path.join(output_dir, base_filename.replace('.csv', '') + '_wordcloud.png')
        plt.figure(figsize=(100, 50))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for {base_filename.replace('.csv', '')}")
        plt.savefig(output_image_path, bbox_inches='tight')
        plt.close()  # Close the figure to avoid displaying it in some environments
        print(f"Word cloud saved to {output_image_path}")

# Function to process all txt files in a folder
def process_folder(folder_path, output_dir, plot):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each txt file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            process_file(file_path, output_dir, plot)


# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Process a folder of .txt files, apply TF-IDF, and save to CSV.")
    parser.add_argument('folder', type=str, help="Folder containing .txt files")
    parser.add_argument('output', type=str, help="Output folder to save CSV files")
    parser.add_argument('--plot', action='store_true', help="Plot word cloud for each file")

    args = parser.parse_args()

    process_folder(args.folder, args.output, args.plot)

if __name__ == "__main__":
    main()
