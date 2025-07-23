import lancedb
import numpy as np
import pandas as pd
from typing import Optional, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

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
    ) -> None:
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
                cell_id = metadata.index[j]
                record = {
                    "vector": embeddings[j].tolist(),
                    "cell_id": cell_id,
                    "embedding_key": embedding_key,
                    "dataset": dataset_name or "unknown",
                    **{col: metadata.iloc[j][col] for col in metadata.columns}
                }
                batch_data.append(record)
            
            data.extend(batch_data)
        
        # Create or append to table
        if self.table_name in self.db.table_names():
            table = self.db.open_table(self.table_name)
            (
                table.merge_insert(["cell_id", "dataset"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(data)
            )
        else:
            self.db.create_table(self.table_name, data=data)
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter: str | None = None,
        include_distance: bool = True,
        columns: List[str] | None = None,
        include_vector: bool = False
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
            query = query.select(columns + ['_distance'] if include_distance else columns)
        
        # convert to pandas
        results = query.to_pandas()

        # deal with _distance column
        if '_distance' in results.columns:
            if include_distance:
                results = results.rename(columns={'_distance': 'query_subject_distance'})
            else:
                results = results.drop('_distance', axis=1)
        elif include_distance:
            results['query_subject_distance'] = 0.0

        # drop vector column if include_vector is False
        if not include_vector and 'vector' in results.columns:
            results = results.drop('vector', axis=1)
        
        return results
    
    def _search_single(self, query_vector: np.ndarray, k: int, filter: str | None, 
                      include_distance: bool, include_vector: bool):
        """Helper method for parallel search."""
        return self.search(
            query_vector=query_vector,
            k=k,
            filter=filter,
            include_distance=include_distance,
            include_vector=include_vector,
        )
    
    def batch_search(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
        filter: str | None = None,
        include_distance: bool = True,
        include_vector: bool = False,
        max_workers: int = 4,
        batch_size: int = 1000,
        show_progress: bool = True,
    ):
        """Parallel batch search for multiple query vectors using ThreadPoolExecutor.
        
        Args:
            query_vectors: Array of query embedding vectors
            k: Number of results per query
            filter: Optional filter expression
            include_distance: Whether to include distances
            include_vector: Whether to include the query vector in the results
            max_workers: Maximum number of worker threads
            batch_size: Number of queries to submit to executor at once
            show_progress: Show progress bar
        Returns:
            List of DataFrames with search results
        """
        from tqdm import tqdm
        
        # Create a partial function with fixed parameters
        search_func = partial(
            self._search_single,
            k=k,
            filter=filter,
            include_distance=include_distance,
            include_vector=include_vector,
        )
        
        results = [None] * len(query_vectors)
        
        # Process in batches to manage memory and avoid overwhelming the database
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            total_processed = 0
            
            if show_progress:
                pbar = tqdm(total=len(query_vectors), desc="Searching")
            
            for batch_start in range(0, len(query_vectors), batch_size):
                batch_end = min(batch_start + batch_size, len(query_vectors))
                batch_vectors = query_vectors[batch_start:batch_end]
                
                # Submit batch to executor
                future_to_index = {
                    executor.submit(search_func, batch_vectors[i]): batch_start + i 
                    for i in range(len(batch_vectors))
                }
                
                # Collect results for this batch
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        print(f"Query {index} failed: {e}")
                        results[index] = pd.DataFrame()  # Empty result on error
                    
                    total_processed += 1
                    if show_progress:
                        pbar.update(1)
            
            if show_progress:
                pbar.close()
        
        return results
    
    def get_table_info(self):
        """Get information about the embeddings table."""
        if self.table_name not in self.db.table_names():
            return None
        
        table = self.db.open_table(self.table_name)
        return {
            "num_rows": len(table),
            "columns": table.schema.names,
            "embedding_dim": len(table.to_pandas().iloc[0]['vector']) if len(table) > 0 else 0
        }

    def get_database_summary(self) -> dict:
        """Get comprehensive summary statistics about the database contents.
        
        Returns:
            Dictionary containing database statistics including:
            - num_cells: Total number of cells stored
            - num_datasets: Number of unique datasets
            - num_embedding_keys: Number of unique embedding keys
            - datasets: List of dataset names
            - embedding_keys: List of embedding key names
            - cells_per_dataset: Dictionary mapping dataset to cell count
        """
        if self.table_name not in self.db.table_names():
            return {
                "num_cells": 0,
                "num_datasets": 0,
                "num_embedding_keys": 0,
                "datasets": [],
                "embedding_keys": [],
                "cells_per_dataset": {},
                "table_exists": False
            }
        
        table = self.db.open_table(self.table_name)
        
        # Get the full dataset to compute statistics
        # For large tables, we might want to optimize this with SQL-like queries
        df = table.to_pandas()
        
        if len(df) == 0:
            return {
                "num_cells": 0,
                "num_datasets": 0,
                "num_embedding_keys": 0,
                "datasets": [],
                "embedding_keys": [],
                "cells_per_dataset": {},
                "table_exists": True
            }
        
        # Calculate summary statistics
        datasets = df['dataset'].unique().tolist()
        embedding_keys = df['embedding_key'].unique().tolist()
        cells_per_dataset = df['dataset'].value_counts().to_dict()
        
        summary = {
            "num_cells": len(df),
            "num_datasets": len(datasets),
            "num_embedding_keys": len(embedding_keys),
            "datasets": sorted(datasets),
            "embedding_keys": sorted(embedding_keys),
            "cells_per_dataset": cells_per_dataset,
            "table_exists": True,
            "embedding_dim": len(df.iloc[0]['vector']) if 'vector' in df.columns else 0
        }
        
        return summary