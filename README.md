# ServiceNow Ticket Clusterer

This project analyzes and clusters ServiceNow tickets based on their short descriptions. It uses a pre-trained sentence transformer model to generate vector embeddings for the ticket descriptions and then uses k-means clustering to group similar tickets together.

## Features

-   Clusters ServiceNow tickets from a CSV file.
-   Generates cluster names based on the most relevant words (TF-IDF).
-   Prints a report with the number of tickets in each cluster and the tickets themselves.

## Requirements

The project requires the following Python libraries:

-   `pandas`
-   `sentence-transformers`
-   `scikit-learn`
-   `numpy`

You can install them using pip:

```bash
pip install -r requirements.txt
```

## Usage

1.  Make sure you have a `servicenow_tickets.csv` file in the same directory as the script, or modify the `csv_file` variable in the `if __name__ == '__main__':` block to point to your CSV file. The CSV file must contain a column named `short_description` with the ticket descriptions.
2.  Run the `ticket_clusterer.py` script:

```bash
python ticket_clusterer.py
```

The script will print the following:

-   A count of tickets in each cluster, along with a generated name for the cluster.
-   A list of the tickets belonging to each cluster.

## Customization

-   **Number of Clusters:** You can change the number of clusters by modifying the `n_clusters` parameter in the `cluster_tickets` function call within the `if __name__ == '__main__':` block.
-   **CSV File and Column:** You can change the input CSV file and the column containing the descriptions by modifying the `csv_file` and `description_column` variables in the `if __name__ == '__main__':` block.
