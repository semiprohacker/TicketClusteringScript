import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib
import os

class TicketClusterer:
    """
    A class to cluster ServiceNow tickets based on their descriptions.
    """
    def __init__(self, n_clusters=12, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the TicketClusterer.

        Args:
            n_clusters (int): The number of clusters to create.
            model_name (str): The name of the sentence transformer model to use.
        """
        self.n_clusters = n_clusters
        self.model = SentenceTransformer(model_name)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.cluster_names = {}

    def fit(self, descriptions):
        """
        Fits the clustering model to the given descriptions.

        Args:
            descriptions (list): A list of ticket descriptions.
        """
        embeddings = self.model.encode(descriptions, show_progress_bar=False)
        self.kmeans.fit(embeddings)
        # Generate cluster names after fitting
        self._generate_cluster_names(descriptions, self.kmeans.labels_)

    def predict(self, descriptions):
        """
        Predicts the cluster for new descriptions.

        Args:
            descriptions (list): A list of new ticket descriptions.

        Returns:
            np.ndarray: The predicted cluster labels.
        """
        embeddings = self.model.encode(descriptions, show_progress_bar=False)
        return self.kmeans.predict(embeddings)

    def fit_predict(self, descriptions):
        """
        Fits the model and predicts the clusters for the given descriptions.

        Args:
            descriptions (list): A list of ticket descriptions.

        Returns:
            np.ndarray: The predicted cluster labels.
        """
        embeddings = self.model.encode(descriptions, show_progress_bar=False)
        labels = self.kmeans.fit_predict(embeddings)
        # Generate cluster names after fitting
        self._generate_cluster_names(descriptions, labels)
        return labels

    def _generate_cluster_names(self, descriptions, labels, n_words=2):
        """Generates names for each cluster based on TF-IDF."""
        df = pd.DataFrame({'description': descriptions, 'cluster': labels})
        docs_per_cluster = df.groupby('cluster')['description'].apply(' '.join)
        
        tfidf_matrix = self.vectorizer.fit_transform(docs_per_cluster)
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        for i, r in enumerate(tfidf_matrix):
            top_word_indices = r.toarray().argsort()[0, -n_words:][::-1]
            top_words = feature_names[top_word_indices]
            self.cluster_names[docs_per_cluster.index[i]] = " - ".join(top_words)

    def save(self, file_path):
        """Saves the clusterer object to a file."""
        joblib.dump(self, file_path)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load(file_path):
        """Loads a clusterer object from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file or directory: '{file_path}'")
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model

if __name__ == '__main__':
    # --- 1. Training and Saving the Model ---
    print("--- Training Model ---")
    excel_file = 'generated_incidents.xlsx'
    try:
        df = pd.read_excel(excel_file)
        descriptions = df['Short Description'].tolist()

        # Initialize and train the clusterer
        clusterer = TicketClusterer(n_clusters=10)
        df['cluster'] = clusterer.fit_predict(descriptions)

        # Save the trained model
        model_path = 'ticket_clusterer_model.joblib'
        clusterer.save(model_path)

        # --- Display Results from Training ---
        print("\n--- Ticket Counts per Cluster (from training) ---")
        counts = df['cluster'].value_counts()
        for cluster_num in sorted(counts.index):
            count = counts[cluster_num]
            name = clusterer.cluster_names.get(cluster_num, "Unnamed Cluster")
            print(f"Cluster {cluster_num} ({name}): {count} tickets")

    except FileNotFoundError:
        print(f"Error: The file '{excel_file}' was not found. Please create it to run the training.")
    except Exception as e:
        print(f"An error occurred during training: {e}")


    # --- 2. Loading the Model and Predicting on New Data ---
    print("\n\n--- Predicting with Loaded Model ---")
    model_path = 'ticket_clusterer_model.joblib'
    if os.path.exists(model_path):
        # Load the saved model
        loaded_clusterer = TicketClusterer.load(model_path)

        # Example of new, unseen data
        new_tickets = pd.read_excel('test_incidents.xlsx',sheet_name='test_data')['Short Description'].tolist()

        # Predict the clusters for the new tickets
        predicted_clusters = loaded_clusterer.predict(new_tickets)

        print("\n--- Predictions for New Tickets ---")
        for ticket, cluster_num in zip(new_tickets, predicted_clusters):
            name = loaded_clusterer.cluster_names.get(cluster_num, "Unnamed Cluster")
            print(f"- '{ticket}' -> Belongs to Cluster {cluster_num} ({name})")
    else:
        print("\nModel file not found. Run the training part of the script first.")
