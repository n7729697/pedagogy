# pedagogy
ATTC1 2024 pegagogy project

## Installation
To set up the environment, ensure the required libraries are installed. You can install them with:
```bash
pip install -r requirements.txt
```

## Preprocessing txt texts
```python
python3 preprocess.py data/text_files output --plot
```
**File Structure**
```
output/
├── example_file.csv           # CSV file with word counts and TF-IDF scores
├── example_file_wordcloud.png  # Optional word cloud image
```
Modify the preprocessing steps (like additional stopwords) by editing the `unwanted_words` set in the script.

## Aspect-Based Sentiment Analysis (ABSA)
Run the script by specifying the path to a folder containing `.txt` files. The output will be saved as a CSV file, with each row representing one text file and columns for sentiment scores of specified aspects.

```python
python3 absa.py
```
1. **Define Aspects**: In the script, adjust the aspects list if you want to analyze different or additional topics.
2. **Specify Folder Path**: Update `folder_path` with the path to your .txt files.

## Rest program
The rest python scripts could be run through `python3 scripts.py`
