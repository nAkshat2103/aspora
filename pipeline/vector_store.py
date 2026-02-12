# """Vector store manager using ChromaDB with persistence."""

# import chromadb
# from sentence_transformers import SentenceTransformer


# class VectorStoreManager:
#     """Manages a persistent ChromaDB collection for document embeddings."""

#     def __init__(
#         self,
#         persist_directory: str = "vector_db",
#         collection_name: str = "doc_qa",
#         embedding_model: str = "all-MiniLM-L6-v2",
#     ):
#         self.persist_directory = persist_directory
#         self.collection_name = collection_name
#         self.embedding_model_name = embedding_model
#         self._model = SentenceTransformer(embedding_model)
#         self._client = chromadb.PersistentClient(path=persist_directory)
#         self._collection = self._client.get_or_create_collection(
#             name=collection_name,
#             metadata={"hnsw:space": "cosine"},
#         )

#     def _embed(self, texts: list[str]) -> list[list[float]]:
#         """Generate embeddings for a list of texts."""
#         return self._model.encode(texts).tolist()

#     def add(
#         self,
#         chunks: list[str],
#         metadata: dict | list[dict] | None = None,
#     ) -> None:
#         """Add text chunks with metadata. metadata can be a single dict (replicated) or list of dicts (one per chunk)."""
#         if not chunks:
#             return
#         if isinstance(metadata, dict):
#             metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]
#         elif isinstance(metadata, list) and len(metadata) == len(chunks):
#             metadatas = [{**m, "chunk_index": i} for i, m in enumerate(metadata)]
#         else:
#             metadatas = [{"chunk_index": i} for i in range(len(chunks))]

#         doc_id = metadatas[0].get("doc_id", "")
#         embeddings = self._embed(chunks)
#         ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
#         self._collection.add(
#             ids=ids,
#             embeddings=embeddings,
#             documents=chunks,
#             metadatas=metadatas,
#         )

#     def search(
#         self,
#         query: str,
#         k: int = 6,
#         fetch_k: int = 20,
#         distance_threshold: float = 0.7,
#         where: dict | None = None,
#     ) -> list[dict]:
#         """
#         Advanced RAG retrieval:
#         - retrieves more candidates (fetch_k)
#         - filters weak matches using distance_threshold
#         - returns best top-k

#         Also guards against hnswlib errors like:
#         "Cannot return the results in a contigious 2D array. Probably ef or M is too small"
#         by capping and, if needed, reducing n_results.
#         """

#         # Embed query
#         query_embedding = self._embed([query])

#         # Make sure we don't ask HNSW for more neighbors than it can safely return.
#         # Clamp to a conservative upper bound to avoid "contiguous 2D array" errors.
#         # Also never ask for more results than we have points in the index.
#         try:
#             total_points = self._collection.count()
#         except Exception:
#             total_points = None

#         max_safe = 5  # very conservative to stay <= typical ef values
#         candidates_cap = [max(1, k), fetch_k, max_safe]
#         if total_points is not None:
#             candidates_cap.append(max(1, total_points))
#         effective_fetch_k = max(1, min(*candidates_cap))

#         try:
#             results = self._collection.query(
#                 query_embeddings=query_embedding,
#                 n_results=effective_fetch_k,
#                 include=["documents", "metadatas", "distances"],
#                 where=where,
#             )
#         except Exception as e:
#             # Fallback: retry with a smaller n_results if HNSW cannot satisfy the request
#             # (e.g. ef or M too small). Match both correct and misspelled forms.
#             msg = str(e).lower()
#             if "contiguous 2d array" in msg or "contigious 2d array" in msg:
#                 fallback_k = 1
#                 results = self._collection.query(
#                     query_embeddings=query_embedding,
#                     n_results=fallback_k,
#                     include=["documents", "metadatas", "distances"],
#                     where=where,
#                 )
#             else:
#                 raise

#         candidates: list[dict] = []

#         if results["documents"] and results["documents"][0]:
#             for i, doc in enumerate(results["documents"][0]):
#                 meta = (results["metadatas"][0][i] or {}) if results["metadatas"] else {}
#                 dist = results["distances"][0][i] if results["distances"] else 0.0

#                 # distance threshold filtering
#                 if dist <= distance_threshold:
#                     candidates.append(
#                         {
#                             "document": doc,
#                             "metadata": meta,
#                             "distance": dist,
#                         }
#                     )

#         # Sort by distance (lower is better)
#         candidates.sort(key=lambda x: x["distance"])

#         # return final top-k
#         return candidates[:k]


#     def delete(self, doc_id: str) -> None:
#         """Delete all chunks belonging to a document."""
#         self._collection.delete(where={"doc_id": doc_id})











# """Vector store manager using ChromaDB with persistence."""

# import chromadb
# from sentence_transformers import SentenceTransformer


# class VectorStoreManager:
#     """Manages a persistent ChromaDB collection for document embeddings."""

#     def __init__(
#         self,
#         persist_directory: str = "vector_db",
#         collection_name: str = "doc_qa",
#         embedding_model: str = "all-MiniLM-L6-v2",
#     ):
#         self.persist_directory = persist_directory
#         self.collection_name = collection_name
#         self.embedding_model_name = embedding_model

#         self._model = SentenceTransformer(embedding_model)
#         self._client = chromadb.PersistentClient(path=persist_directory)

#         # IMPORTANT: Set proper HNSW params (requires re-ingest after change)
#         self._collection = self._client.get_or_create_collection(
#             name=collection_name,
#             metadata={
#                 "hnsw:space": "cosine",
#                 "hnsw:ef_search": 64,
#                 "hnsw:M": 64,
#             },
#         )

#     def _embed(self, texts: list[str]) -> list[list[float]]:
#         """Generate embeddings for a list of texts."""
#         return self._model.encode(texts).tolist()

#     def add(
#         self,
#         chunks: list[str],
#         metadata: dict | list[dict] | None = None,
#     ) -> None:
#         """Add text chunks with metadata."""
#         if not chunks:
#             return

#         if isinstance(metadata, dict):
#             metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]
#         elif isinstance(metadata, list) and len(metadata) == len(chunks):
#             metadatas = [{**m, "chunk_index": i} for i, m in enumerate(metadata)]
#         else:
#             metadatas = [{"chunk_index": i} for i in range(len(chunks))]

#         doc_id = metadatas[0].get("doc_id", "")
#         embeddings = self._embed(chunks)
#         ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

#         self._collection.add(
#             ids=ids,
#             embeddings=embeddings,
#             documents=chunks,
#             metadatas=metadatas,
#         )

#     def search(
#         self,
#         query: str,
#         k: int = 6,
#         fetch_k: int = 20,
#         distance_threshold: float | None = None,
#         where: dict | None = None,
#     ) -> list[dict]:
#         """
#         Advanced RAG retrieval:
#         - retrieves up to fetch_k candidates
#         - optionally filters weak matches (if distance_threshold is set)
#         - returns best top-k

#         By default, **no distance threshold** is applied so that we always
#         return something when any chunks exist in the store.
#         """

#         query_embedding = self._embed([query])

#         # Determine a safe upper bound for n_results.
#         try:
#             total_points = self._collection.count()
#         except Exception:
#             total_points = None

#         n_results = fetch_k
#         if total_points:
#             n_results = min(fetch_k, total_points)

#         # Robust query: if hnswlib complains about ef/M vs n_results, back off n_results.
#         while True:
#             try:
#                 results = self._collection.query(
#                     query_embeddings=query_embedding,
#                     n_results=n_results,
#                     include=["documents", "metadatas", "distances"],
#                     where=where,
#                 )
#                 break
#             except Exception as e:
#                 msg = str(e).lower()
#                 if (
#                     ("contiguous 2d array" in msg or "contigious 2d array" in msg)
#                     and n_results > 1
#                 ):
#                     # Back off n_results and try again.
#                     n_results = max(1, n_results // 2)
#                     continue
#                 # Any other error, or already at 1, propagate.
#                 raise

#         candidates: list[dict] = []

#         if results["documents"] and results["documents"][0]:
#             for i, doc in enumerate(results["documents"][0]):
#                 meta = (results["metadatas"][0][i] or {}) if results["metadatas"] else {}
#                 dist = results["distances"][0][i] if results["distances"] else 0.0

#                 # If no threshold is provided, accept all candidates.
#                 if distance_threshold is None or dist <= distance_threshold:
#                     candidates.append(
#                         {
#                             "document": doc,
#                             "metadata": meta,
#                             "distance": dist,
#                         }
#                     )

#         candidates.sort(key=lambda x: x["distance"])
#         return candidates[:k]

#     def get_chunks_by_indices(
#         self,
#         doc_id: str,
#         indices: list[int],
#     ) -> list[dict]:
#         """
#         Fetch specific chunks for a document by their chunk_index values.

#         Returns a list of dicts with the same shape as search() results:
#         {"document": str, "metadata": dict}
#         """
#         # Deduplicate and keep only non-negative indices.
#         unique_indices = sorted({i for i in indices if i is not None and i >= 0})
#         if not unique_indices:
#             return []

#         ids = [f"{doc_id}_{i}" for i in unique_indices]
#         results = self._collection.get(ids=ids, include=["documents", "metadatas"])

#         documents = results.get("documents") or []
#         metadatas = results.get("metadatas") or []

#         out: list[dict] = []
#         for i, doc in enumerate(documents):
#             if doc is None:
#                 continue
#             meta = metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
#             out.append(
#                 {
#                     "document": doc,
#                     "metadata": meta,
#                 }
#             )
#         return out

#     def delete(self, doc_id: str) -> None:
#         """Delete all chunks belonging to a document."""
#         self._collection.delete(where={"doc_id": doc_id})





























"""Vector store manager using ChromaDB with persistence (ID-safe)."""

import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization for BM25."""
    return re.findall(r"\w+", text.lower())


class VectorStoreManager:
    """Manages a persistent ChromaDB collection for document embeddings."""

    def __init__(
        self,
        persist_directory: str = "vector_db",
        collection_name: str = "doc_qa",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        self._model = SentenceTransformer(embedding_model)
        self._client = chromadb.PersistentClient(path=persist_directory)

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []
        self._rebuild_bm25()

    # ---------------------------------------------------------
    # Internal
    # ---------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from Chroma collection."""
        try:
            results = self._collection.get(include=["documents"])
        except (StopIteration, Exception):
            self._bm25 = None
            self._bm25_docs = []
            return

        ids_list = results.get("ids") or []
        docs_list = results.get("documents") or []
        metas_list = results.get("metadatas") or []

        self._bm25_docs = []
        for i, doc in enumerate(docs_list):
            if doc is None:
                continue
            meta = metas_list[i] if i < len(metas_list) and metas_list[i] else {}
            self._bm25_docs.append({
                "id": ids_list[i] if i < len(ids_list) else f"chunk_{i}",
                "document": doc,
                "metadata": meta,
            })

        if not self._bm25_docs:
            self._bm25 = None
            return

        tokenized = [_tokenize(d["document"]) for d in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)

    def _recreate_collection(self) -> None:
        """Recreate collection (workaround for Chroma StopIteration when collection has no segments)."""
        try:
            self._client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _delete_by_doc_id_prefix(self, doc_id: str) -> None:
        try:
            # Chroma get() only accepts include=["documents","embeddings","metadatas","distances","uris","data"] - not "ids"
            results = self._collection.get(include=["documents"])
        except (StopIteration, Exception):
            return
        existing_ids = results.get("ids") or []
        to_delete = [i for i in existing_ids if i.startswith(f"{doc_id}_")]
        if to_delete:
            try:
                self._collection.delete(ids=to_delete)
            except (StopIteration, Exception):
                pass


    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def add(
        self,
        chunks: list[str],
        metadata: dict | list[dict] | None = None,
    ) -> None:
        """Add text chunks with metadata (safe against ID collisions)."""

        if not chunks:
            return

        if isinstance(metadata, dict):
            metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]
        elif isinstance(metadata, list) and len(metadata) == len(chunks):
            metadatas = [{**m, "chunk_index": i} for i, m in enumerate(metadata)]
        else:
            metadatas = [{"chunk_index": i} for i in range(len(chunks))]

        doc_id = metadatas[0].get("doc_id")
        if not doc_id:
            raise ValueError("doc_id is required in metadata.")

        # Prevent ID collision by removing old chunks first
        self._delete_by_doc_id_prefix(doc_id)

        embeddings = self._embed(chunks)
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )
        except StopIteration:
            # Chroma raises StopIteration when collection has no segments (empty/corrupt state)
            self._recreate_collection()
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

        self._rebuild_bm25()

    def search(
        self,
        query: str,
        k: int = 6,
        fetch_k: int = 20,
        distance_threshold: float | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Hybrid retrieval: vector similarity + BM25, merged with Reciprocal Rank Fusion.
        """
        query_embedding = self._embed([query])

        # ----- 1. Vector search -----
        try:
            total_points = self._collection.count()
        except Exception:
            total_points = None

        n_results = fetch_k
        if total_points:
            n_results = min(fetch_k, total_points)

        vector_results: list[dict] = []
        try:
            while True:
                try:
                    results = self._collection.query(
                        query_embeddings=query_embedding,
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"],
                        where=where,
                    )
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if (
                        ("contiguous 2d array" in msg or "contigious 2d array" in msg)
                        and n_results > 1
                    ):
                        n_results = max(1, n_results // 2)
                        continue
                    raise

            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = (results["metadatas"][0][i] or {}) if results["metadatas"] else {}
                    dist = results["distances"][0][i] if results["distances"] else 0.0
                    if distance_threshold is None or dist <= distance_threshold:
                        vector_results.append({
                            "document": doc,
                            "metadata": meta,
                            "distance": dist,
                        })
        except Exception:
            pass

        # ----- 2. BM25 search -----
        bm25_results: list[dict] = []
        if self._bm25 and self._bm25_docs:
            try:
                tokenized_q = _tokenize(query)
                if tokenized_q:
                    scores = self._bm25.get_scores(tokenized_q)
                    ranked = sorted(
                        range(len(scores)),
                        key=lambda i: scores[i],
                        reverse=True,
                    )[:fetch_k]
                    for idx in ranked:
                        if scores[idx] > 0:
                            d = self._bm25_docs[idx]
                            bm25_results.append({
                                "document": d["document"],
                                "metadata": d["metadata"],
                                "distance": None,
                            })
            except Exception:
                pass

        # ----- 3. Reciprocal Rank Fusion -----
        RRF_K = 60
        fusion_scores: dict[str, float] = {}

        for rank, item in enumerate(vector_results):
            key = item["document"]
            fusion_scores[key] = fusion_scores.get(key, 0) + 1.0 / (RRF_K + rank)

        for rank, item in enumerate(bm25_results):
            key = item["document"]
            fusion_scores[key] = fusion_scores.get(key, 0) + 1.0 / (RRF_K + rank)

        # ----- 4. Merge and rank -----
        merged: dict[str, dict] = {}
        for item in vector_results:
            merged[item["document"]] = item
        for item in bm25_results:
            if item["document"] not in merged:
                merged[item["document"]] = item

        ranked = sorted(
            merged.values(),
            key=lambda x: fusion_scores.get(x["document"], 0),
            reverse=True,
        )
        return ranked[:k]

    def get_chunks_by_indices(
        self,
        doc_id: str,
        indices: list[int],
    ) -> list[dict]:

        unique_indices = sorted({i for i in indices if i is not None and i >= 0})
        if not unique_indices:
            return []

        ids = [f"{doc_id}_{i}" for i in unique_indices]

        results = self._collection.get(ids=ids, include=["documents", "metadatas"])

        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        out: list[dict] = []
        for i, doc in enumerate(documents):
            if doc is None:
                continue
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
            out.append(
                {
                    "document": doc,
                    "metadata": meta,
                }
            )

        return out

    def delete(self, doc_id: str) -> None:
        """Delete all chunks belonging to a document (safe version)."""
        self._delete_by_doc_id_prefix(doc_id)
