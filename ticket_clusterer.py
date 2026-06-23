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
        self._generate_cluster_names(descriptions, labels,5)
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

def train_model(excel_file, status_callback=print, n_clusters=15):
    """
    Trains the ticket clusterer model and saves it.
    """
    status_callback("--- Training Model ---")
    try:
        df = pd.read_excel(excel_file)
        descriptions = df['Short Description'].tolist()

        # Initialize and train the clusterer
        clusterer = TicketClusterer(n_clusters=n_clusters)
        df['cluster'] = clusterer.fit_predict(descriptions)

        # Add cluster names to the dataframe
        df['cluster_name'] = df['cluster'].apply(lambda x: clusterer.cluster_names.get(x, "Unnamed Cluster"))

        # Save the trained model
        model_path = 'ticket_clusterer_model.joblib'
        clusterer.save(model_path)
        status_callback(f"Model saved to {model_path}")

        # --- Save Training Results to Excel ---
        status_callback("--- Saving training results to Excel ---")
        training_output_file = 'training_cluster_analysis.xlsx'
        
        # Create cluster counts summary
        cluster_counts_df = df.groupby(['cluster', 'cluster_name']).size().reset_index(name='ticket_count')
        cluster_counts_df = cluster_counts_df.sort_values('cluster').reset_index(drop=True)

        with pd.ExcelWriter(training_output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Ticket Clusters', index=False)
            cluster_counts_df.to_excel(writer, sheet_name='Cluster Counts', index=False)
        
        status_callback(f"Training analysis saved to '{training_output_file}'")
        return True, "Training complete."

    except FileNotFoundError:
        error_msg = f"Error: The file '{excel_file}' was not found. Please create it to run the training."
        status_callback(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"An error occurred during training: {e}"
        status_callback(error_msg)
        return False, error_msg

def predict_with_model(excel_file, status_callback=print):
    """
    Loads a trained model and predicts on new data.
    """
    status_callback("--- Predicting with Loaded Model ---")
    model_path = 'ticket_clusterer_model.joblib'
    if os.path.exists(model_path):
        try:
            # Load the saved model
            loaded_clusterer = TicketClusterer.load(model_path)
            status_callback(f"Model loaded from {model_path}")

            # Example of new, unseen data
            new_tickets = pd.read_excel(excel_file,sheet_name='test_data')['Short Description'].tolist()

            # Predict the clusters for the new tickets
            predicted_clusters = loaded_clusterer.predict(new_tickets)

            # --- Save Prediction Results to Excel ---
            status_callback("--- Saving prediction results to Excel ---")
            prediction_output_file = 'prediction_cluster_analysis.xlsx'

            # Create a DataFrame for the new predictions
            predictions_df = pd.DataFrame({
                'Short Description': new_tickets,
                'predicted_cluster': predicted_clusters
            })
            predictions_df['predicted_cluster_name'] = predictions_df['predicted_cluster'].apply(lambda x: loaded_clusterer.cluster_names.get(x, "Unnamed Cluster"))

            # Create cluster counts summary for predictions
            pred_cluster_counts_df = predictions_df.groupby(['predicted_cluster', 'predicted_cluster_name']).size().reset_index(name='ticket_count')
            pred_cluster_counts_df = pred_cluster_counts_df.sort_values('predicted_cluster').reset_index(drop=True)

            with pd.ExcelWriter(prediction_output_file, engine='openpyxl') as writer:
                predictions_df.to_excel(writer, sheet_name='Predicted Ticket Clusters', index=False)
                pred_cluster_counts_df.to_excel(writer, sheet_name='Predicted Cluster Counts', index=False)

            status_callback(f"Prediction analysis saved to '{prediction_output_file}'")
            return True, "Prediction complete."
        
        except Exception as e:
            error_msg = f"An error occurred during prediction: {e}"
            status_callback(error_msg)
            return False, error_msg
    else:
        error_msg = "Model file not found. Run the training part of the script first."
        status_callback(error_msg)
        return False, error_msg

if __name__ == '__main__':
    train_model('generated_incidents.xlsx', print)
    predict_with_model('test_incidents.xlsx', print)
