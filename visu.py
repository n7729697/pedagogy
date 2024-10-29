import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#plt.style.use('seaborn-darkgrid')
# Set the font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# Load the CSV results into a DataFrame
df = pd.read_csv('/home/xuezhi/Desktop/aspect_sentiment_analysis_results.csv')
# Sort data by rank
df = df.sort_values(by='name')
# Display the first few rows to verify the data
print(df.head())

# Set up some common parameters for the plots
plt.style.use('seaborn-v0_8-darkgrid')
figsize = (12, 6)

### Bar Chart: Comparing Sentiment Across Aspects ###
def plot_bar_chart(df):
    df.set_index('name').plot(kind='bar', figsize=figsize)
    plt.title('Aspect Sentiment Scores for Each File')
    plt.xlabel('File')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.legend(title='Aspects', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

### Heatmap: Visualize Sentiment Across Files and Aspects ###
def plot_heatmap(df):
    # Group by 'Location' and calculate the mean of each aspect
    #print(df.'name')
    location_means = df.groupby('Location').mean()[['engage', 'privacy', 'access', 'teaching', 'learning']]
    print(location_means)
    plt.figure(figsize=figsize)
    sns.heatmap(location_means, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Heatmap of Aspect Scores by Location')
    plt.ylabel('Location')
    plt.xlabel('Aspect')
    plt.show()

### Line Plot: Trends Across Files ###
def plot_line_chart(df):
    plt.figure(figsize=figsize)
    for aspect in df.columns[1:6]:  # Skip 'file' column
        plt.plot(df['name'], df[aspect], marker='o', label=aspect)

    plt.title('Sentiment Trends for Each Aspect Across Swedish Universities')
    plt.xlabel('University Name')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.legend(title='Aspects')
    plt.tight_layout()
    plt.show()

### Main Function to Run All Plots ###
def main():
    # Plot bar chart
    #print("Plotting bar chart...")
    #plot_bar_chart(df)
    
    # Plot heatmap
    #print("Plotting heatmap...")
    #plot_heatmap(df)
    
    # Plot line chart
    print("Plotting line chart...")
    plot_line_chart(df)

# Execute the main function
if __name__ == "__main__":
    main()

