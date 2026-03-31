import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def cluster_tickets(csv_path, description_column='short_description', n_clusters=5):
    """
    Clusters tickets from a CSV file based on their descriptions.

    Args:
        csv_path (str): The path to the CSV file.
        description_column (str): The name of the column containing the ticket descriptions.
        n_clusters (int): The number of clusters to create.

    Returns:
        pd.DataFrame: A pandas DataFrame with the original data and a new 'cluster' column.
    """
    # Load the data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None

    if description_column not in df.columns:
        print(f"Error: The CSV file must have a '{description_column}' column.")
        return None

    # Load a pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the ticket descriptions
    descriptions = df[description_column].tolist()
    embeddings = model.encode(descriptions, show_progress_bar=False)

    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(embeddings)

    return df

def generate_cluster_names(clustered_df, n_words=2):
    """Generates names for each cluster based on TF-IDF."""
    # Combine descriptions for each cluster into a single string
    docs_per_cluster = clustered_df.groupby('cluster')['short_description'].apply(' '.join)

    # Use TF-IDF to find important words
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(docs_per_cluster)
    
    # Get feature names (the words)
    feature_names = np.array(vectorizer.get_feature_names_out())

    cluster_names = {}
    for i, r in enumerate(tfidf_matrix):
        # Get the indices of the top n_words for this cluster
        top_word_indices = r.toarray().argsort()[0, -n_words:][::-1]
        top_words = feature_names[top_word_indices]
        cluster_names[docs_per_cluster.index[i]] = " - ".join(top_words)

    return cluster_names


if __name__ == '__main__':
    csv_file = 'servicenow_tickets.csv'
    # I've chosen 4 clusters based on the sample data categories (login, performance, payment, booking).
    # You can change this number to see how it affects the clustering.
    clustered_df = cluster_tickets(csv_file, n_clusters=4)

    if clustered_df is not None:
        # Generate cluster names
        cluster_names = generate_cluster_names(clustered_df, n_words=2)

        print("--- Ticket Counts per Cluster ---")
        # Print counts with names
        counts = clustered_df['cluster'].value_counts()
        for cluster_num in counts.index:
            count = counts[cluster_num]
            name = cluster_names.get(cluster_num, "Unnamed Cluster")
            print(f"Cluster {cluster_num} ({name}): {count} tickets")

        print("\n--- Tickets in each Cluster ---")
        for cluster_num in sorted(clustered_df['cluster'].unique()):
            name = cluster_names.get(cluster_num, "Unnamed Cluster")
            print(f"\n--- Cluster {cluster_num}: {name} ---")
            cluster_tickets = clustered_df[clustered_df['cluster'] == cluster_num]['short_description'].tolist()
            for ticket in cluster_tickets:
                print(f"- {ticket}")
