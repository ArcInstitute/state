import lancedb
import numpy as np
import pandas as pd
from typing import Optional, List


class StateVectorDB:
    """Manages LanceDB operations for State embeddings."""

    def __init__(self, db_path: str = "./state_embeddings.lancedb"):
        """Initialize or connect to a LanceDB database.

        Args:
            db_path: Path to the LanceDB database
        """
        self.db = lancedb.connect(db_path)
        self.table_name = "state_embeddings"

    def create_or_update_table(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        embedding_key: str = "X_state",
        dataset_name: Optional[str] = None,
        batch_size: int = 1000,
    ):
        """Create or update the embeddings table.

        Args:
            embeddings: Cell embeddings array (n_cells x embedding_dim)
            metadata: Cell metadata from adata.obs
            embedding_key: Name of the embedding (for versioning)
            dataset_name: Name of the dataset being processed
            batch_size: Batch size for insertion
        """
        # Prepare data with metadata
        data = []
        for i in range(0, len(embeddings), batch_size):
            batch_end = min(i + batch_size, len(embeddings))
            batch_data = []

            for j in range(i, batch_end):
                record = {
                    "vector": embeddings[j].tolist(),
                    "cell_id": metadata.index[j],
                    "embedding_key": embedding_key,
                    "dataset": dataset_name or "unknown",
                    **{col: metadata.iloc[j][col] for col in metadata.columns},
                }
                batch_data.append(record)

            data.extend(batch_data)

        # Create or append to table
        if self.table_name in self.db.table_names():
            table = self.db.open_table(self.table_name)
            table.add(data)
        else:
            self.db.create_table(self.table_name, data=data)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: str | None = None,
        include_distance: bool = True,
        columns: List[str] | None = None,
        include_vector: bool = False,
    ):
        """Search for similar embeddings.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filter: Optional filter expression (e.g., 'cell_type == "B cell"')
            include_distance: Whether to include distance in results
            include_vector: Whether to include the query vector in the results
            columns: Specific columns to return (None = all)
        Returns:
            Search results with metadata
        """
        table = self.db.open_table(self.table_name)

        # Build query
        query = table.search(query_vector).limit(k)

        if filter:
            query = query.where(filter)

        if columns:
            query = query.select(columns + ["_distance"] if include_distance else columns)

        results = query.to_pandas()

        # deal with _distance column
        if "_distance" in results.columns:
            if include_distance:
                results = results.rename(columns={"_distance": "query_distance"})
            else:
                results = results.drop("_distance", axis=1)
        elif include_distance:
            results["query_distance"] = 0.0

        # drop vector column if include_vector is False
        if not include_vector and "vector" in results.columns:
            results = results.drop("vector", axis=1)

        return results

    def batch_search(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        filter: str | None = None,
        include_distance: bool = True,
        batch_size: int = 100,
        show_progress: bool = True,
        include_vector: bool = False,
    ):
        """Batch search for multiple query vectors.

        Args:
            query_vectors: Array of query embedding vectors
            k: Number of results per query
            filter: Optional filter expression
            include_distance: Whether to include distances
            include_vector: Whether to include the query vector in the results
            batch_size: Number of queries to process at once
            show_progress: Show progress bar
        Returns:
            List of DataFrames with search results
        """
        from tqdm import tqdm

        results = []
        iterator = range(0, len(query_vectors), batch_size)

        if show_progress:
            iterator = tqdm(iterator, desc="Searching")

        for i in iterator:
            batch_end = min(i + batch_size, len(query_vectors))
            batch_queries = query_vectors[i:batch_end]

            batch_results = []
            for query_vec in batch_queries:
                result = self.search(
                    query_vector=query_vec,
                    k=k,
                    filter=filter,
                    include_distance=include_distance,
                    include_vector=include_vector,
                )
                batch_results.append(result)

            results.extend(batch_results)

        return results

    def get_table_info(self):
        """Get information about the embeddings table."""
        if self.table_name not in self.db.table_names():
            return None

        table = self.db.open_table(self.table_name)
        return {
            "num_rows": len(table),
            "columns": table.schema.names,
            "embedding_dim": len(table.to_pandas().iloc[0]["vector"]) if len(table) > 0 else 0,
        }
