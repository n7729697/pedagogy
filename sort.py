import pandas as pd
import os

# Example csv for processing
folder_path = '/your/csv/files/path'  # Path to your .csv file

# Initialize a new DataFrame to store results
results = pd.DataFrame(columns=['Filename'] + [f'Term_{i+1}' for i in range(10)])

# Process each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):  # Process only .txt files
        file_path = os.path.join(folder_path, filename)
        
        df = pd.read_csv(file_path)
        df_sorted = df.sort_values(by='Word Count', ascending=False)
    	
    	# Select the top 10 terms
        top_10_terms = df_sorted.head(10)['Term'].tolist()
	
	# Create a result row with the filename (without .csv) and the top 10 terms
        result = {'University Name': filename.replace('.csv', '')}
        result.update({f'Term_{i+1}': term for i, term in enumerate(top_10_terms)})
        
        # Append the result to the DataFrame using pd.concat
        results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)

# Save the results to a new CSV file
output_file_path = '/home/xuezhi/Desktop/Top_10_Terms_Summary.csv'
results.to_csv(output_file_path, index=False)

print(f'The summary has been saved to {output_file_path}')
