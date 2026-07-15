import logging
import uuid
import hashlib
from typing import List, Dict, Any
from app.core.config import settings
from app.core.exceptions import DocumentNotIndexedException, ExternalServiceException

logger = logging.getLogger("app.services.vector_store")

# Lazy import statement to prevent circular dependencies or PyTorch/Chroma load overhead
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

class VectorStoreService:
    """
    Manages isolated document collection vector storages inside ChromaDB.
    """
    def __init__(self):
        self._client = None

    def _get_client(self):
        try:
            import chromadb
        except ImportError:
            raise RuntimeError("chromadb library is not installed in the virtual environment.")
        
        if self._client is None:
            logger.info(f"Initializing persistent ChromaDB client at: '{settings.CHROMA_PERSIST_DIR}'")
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=chromadb.Settings(anonymized_telemetry=False)
            )
        return self._client

    def get_collection_name(self, doc_id: uuid.UUID | str) -> str:
        """
        Generates a deterministic collection name using the MD5 hash of the document ID.
        Avoids file collision across name matches.
        """
        doc_str = str(doc_id)
        return f"col_{hashlib.md5(doc_str.encode('utf-8')).hexdigest()}"

# OLD: stored document chunks with only page_number, chunk_index, and source_document_id metadata — replaced below to add was_multi_column and had_tables flags to ChromaDB metadata
#     def store_document_chunks(
#         self,
#         doc_id: uuid.UUID,
#         chunks: List[Dict[str, Any]],
#         embeddings: List[List[float]]
#     ) -> str:
#         """
#         Creates a dedicated collection for a document and adds chunk strings, embeddings, and page metadata.
#         """
#         if not chunks:
#             return ""
# 
#         client = self._get_client()
#         collection_name = self.get_collection_name(doc_id)
#         logger.info(f"Storing chunks in ChromaDB collection: '{collection_name}' for doc ID '{doc_id}'")
# 
#         try:
#             # Delete collection if it already exists to guarantee a clean override
#             try:
#                 client.delete_collection(name=collection_name)
#             except Exception:
#                 pass
# 
#             collection = client.create_collection(
#                 name=collection_name,
#                 metadata={"hnsw:space": "cosine"}
#             )
# 
#             ids = []
#             documents = []
#             metadatas = []
# 
#             for chunk, emb in zip(chunks, embeddings):
#                 chunk_idx = chunk["chunk_index"]
#                 ids.append(f"{doc_id}_chunk_{chunk_idx}")
#                 documents.append(chunk["text"])
#                 metadatas.append({
#                     "page_number": chunk["page_number"],
#                     "chunk_index": chunk_idx,
#                     "source_document_id": str(doc_id)
#                 })
# 
#             collection.add(
#                 ids=ids,
#                 documents=documents,
#                 embeddings=embeddings,
#                 metadatas=metadatas
#             )
#             logger.info(f"Successfully indexed {len(chunks)} chunks in collection: '{collection_name}'")
#             return collection_name
# 
#         except Exception as e:
#             logger.error(f"Failed to write chunks to ChromaDB collection '{collection_name}': {e}")
#             raise ExternalServiceException(f"Failed to write vectors to the database: {str(e)}")

    def store_document_chunks(
        self,
        doc_id: uuid.UUID,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> str:
        """
        Creates a dedicated collection for a document and adds chunk strings, embeddings, and page metadata (including columns and tables flags).
        """
        if not chunks:
            return ""

        client = self._get_client()
        collection_name = self.get_collection_name(doc_id)
        logger.info(f"Storing chunks in ChromaDB collection: '{collection_name}' for doc ID '{doc_id}'")

        try:
            # Delete collection if it already exists to guarantee a clean override
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass

            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            # OLD: ChromaDB stored and indexed only child-chunk-level text and 
            # embeddings — replaced/extended below to also store parent chunk text 
            # (unembedded, just as retrievable metadata/a separate lookup) alongside 
            # child chunk embeddings.
            # Design Decision: We chose to store the full parent chunk text directly as metadata in 
            # each child chunk's ChromaDB entry because it is simple, requires no secondary storage 
            # file/database setup, and scales natively with ChromaDB.
            ids = []
            documents = []
            metadatas = []

            for chunk, emb in zip(chunks, embeddings):
                chunk_idx = chunk["chunk_index"]
                ids.append(f"{doc_id}_chunk_{chunk_idx}")
                documents.append(chunk["text"])
                
                meta = {
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk_idx,
                    "source_document_id": str(doc_id),
                    "was_multi_column": chunk.get("was_multi_column", False),
                    "had_tables": chunk.get("had_tables", False)
                }
                
                if "parent_chunk_id" in chunk:
                    meta["parent_chunk_id"] = chunk["parent_chunk_id"]
                if "parent_chunk_text" in chunk:
                    meta["parent_chunk_text"] = chunk["parent_chunk_text"]
                if "parent_page_range" in chunk:
                    meta["parent_page_range"] = chunk["parent_page_range"]
                    
                metadatas.append(meta)

            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Successfully indexed {len(chunks)} chunks in collection: '{collection_name}'")
            return collection_name

        except Exception as e:
            logger.error(f"Failed to write chunks to ChromaDB collection '{collection_name}': {e}")
            raise ExternalServiceException(f"Failed to write vectors to the database: {str(e)}")


    # OLD: query_similar_chunks did pure vector/cosine similarity search only, 
    # no keyword-based matching — replaced below with a hybrid approach 
    # combining vector search and BM25, fused via Reciprocal Rank Fusion, for 
    # better recall on queries with specific keywords/names/numbers that 
    # embeddings alone sometimes miss.
    # def query_similar_chunks(
    #     self,
    #     query_embedding: List[float],
    #     doc_id: uuid.UUID,
    #     top_k: int = 5
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Queries similarity matches on a document collection.
    #     Translates cosine distance back to a standard similarity score.
    #     """
    #     client = self._get_client()
    #     collection_name = self.get_collection_name(doc_id)
    #     try:
    #         collection = client.get_collection(name=collection_name)
    #     except Exception:
    #         raise DocumentNotIndexedException(
    #             f"The vector storage collection for document '{doc_id}' is missing or has been deleted."
    #         )
    #     try:
    #         results = collection.query(
    #             query_embeddings=[query_embedding],
    #             n_results=top_k
    #         )
    #         matched_chunks = []
    #         if not results or not results.get("ids") or len(results["ids"][0]) == 0:
    #             return []
    #         ids = results["ids"][0]
    #         docs = results["documents"][0]
    #         metadatas = results["metadatas"][0]
    #         distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)
    #         for cid, doc, meta, dist in zip(ids, docs, metadatas, distances):
    #             matched_chunks.append({
    #                 "id": cid,
    #                 "text": doc,
    #                 "page_number": meta.get("page_number"),
    #                 "chunk_index": meta.get("chunk_index"),
    #                 "source_document_id": meta.get("source_document_id"),
    #                 "similarity_score": 1.0 - dist
    #             })
    #         return matched_chunks
    #     except Exception as e:
    #         logger.error(f"Failed to query ChromaDB collection '{collection_name}': {e}")
    #         raise ExternalServiceException(f"Failed to query vector database search: {str(e)}")

    def query_similar_chunks(
        self,
        query_embedding: List[float],
        doc_id: uuid.UUID | str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Deprecated: use query_hybrid instead. Legacy compatibility wrapper.
        """
        return self._query_vector_similarity(query_embedding, doc_id, top_k)

    def _query_vector_similarity(
        self,
        query_embedding: List[float],
        doc_id: uuid.UUID | str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Internal helper performing pure vector/cosine similarity search.
        """
        client = self._get_client()
        collection_name = self.get_collection_name(doc_id)

        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            raise DocumentNotIndexedException(
                f"The vector storage collection for document '{doc_id}' is missing or has been deleted."
            )

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            matched_chunks = []
            if not results or not results.get("ids") or len(results["ids"][0]) == 0:
                return []

            ids = results["ids"][0]
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)

            for cid, doc, meta, dist in zip(ids, docs, metadatas, distances):
                matched_chunks.append({
                    "id": cid,
                    "text": doc,
                    "page_number": meta.get("page_number"),
                    "chunk_index": meta.get("chunk_index"),
                    "source_document_id": meta.get("source_document_id"),
                    "similarity_score": 1.0 - dist,
                    "parent_chunk_id": meta.get("parent_chunk_id"),
                    "parent_chunk_text": meta.get("parent_chunk_text"),
                    "parent_page_range": meta.get("parent_page_range")
                })

            return matched_chunks

        except Exception as e:
            logger.error(f"Failed to query ChromaDB collection '{collection_name}': {e}")
            raise ExternalServiceException(f"Failed to query vector database search: {str(e)}")

    def query_hybrid(
        self,
        query_text: str,
        query_embedding: List[float],
        collection_id: uuid.UUID | str,
        vector_top_k: int = 15,
        bm25_top_k: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Runs vector similarity search AND a BM25 keyword search over the same document's chunks,
        then fuses them via weighted Reciprocal Rank Fusion (RRF).
        """
        # 1. Runs vector similarity search
        vector_chunks = self._query_vector_similarity(
            query_embedding=query_embedding,
            doc_id=collection_id,
            top_k=vector_top_k
        )

        # 2. Runs BM25 keyword search
        client = self._get_client()
        collection_name = self.get_collection_name(collection_id)
        
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            raise DocumentNotIndexedException(
                f"The vector storage collection for document '{collection_id}' is missing or has been deleted."
            )

        try:
            # Deliberate simplicity choice: rebuild the BM25 index in-memory from the chunk texts
            # already stored in ChromaDB to avoid extra storage or synchronization complexity.
            all_chunks = collection.get()
        except Exception as e:
            logger.error(f"Failed to retrieve chunks for BM25 rebuild from collection '{collection_name}': {e}")
            raise ExternalServiceException(f"Failed to retrieve chunks for hybrid search: {str(e)}")

        ids = all_chunks.get("ids", [])
        documents = all_chunks.get("documents", [])
        metadatas = all_chunks.get("metadatas", [])

        if not documents:
            # If no chunks exist, return vector search results (likely empty)
            return vector_chunks

        # Tokenize chunks for BM25
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise RuntimeError("rank-bm25 library is not installed in the virtual environment.")

        # Simple word splitting for tokenizer
        tokenized_corpus = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)

        # Score query
        tokenized_query = query_text.lower().split()
        scores = bm25.get_scores(tokenized_query)

        scored_docs = []
        for idx, (cid, doc, meta, score) in enumerate(zip(ids, documents, metadatas, scores)):
            scored_docs.append({
                "id": cid,
                "text": doc,
                "page_number": meta.get("page_number"),
                "chunk_index": meta.get("chunk_index"),
                "source_document_id": meta.get("source_document_id"),
                "bm25_score": score,
                "parent_chunk_id": meta.get("parent_chunk_id"),
                "parent_chunk_text": meta.get("parent_chunk_text"),
                "parent_page_range": meta.get("parent_page_range")
            })

        # Sort BM25 results descending and slice top K
        ranked_bm25 = sorted(scored_docs, key=lambda x: x["bm25_score"], reverse=True)
        ranked_bm25 = ranked_bm25[:bm25_top_k]

        # 3. Reciprocal Rank Fusion (RRF)
        K_RRF = 60
        
        # Maps from chunk ID to 1-based rank
        vector_ranks = {item["id"]: idx + 1 for idx, item in enumerate(vector_chunks)}
        bm25_ranks = {item["id"]: idx + 1 for idx, item in enumerate(ranked_bm25)}

        # All unique chunk IDs across both searches
        all_chunk_ids = set(vector_ranks.keys()).union(bm25_ranks.keys())

        # Map to quickly fetch metadata
        chunk_metadata = {}
        for item in vector_chunks:
            chunk_metadata[item["id"]] = {
                "chunk_text": item["text"],
                "page_number": item["page_number"],
                "chunk_index": item["chunk_index"],
                "parent_chunk_id": item.get("parent_chunk_id"),
                "parent_chunk_text": item.get("parent_chunk_text"),
                "parent_page_range": item.get("parent_page_range")
            }
        for item in ranked_bm25:
            chunk_metadata[item["id"]] = {
                "chunk_text": item["text"],
                "page_number": item["page_number"],
                "chunk_index": item["chunk_index"],
                "parent_chunk_id": item.get("parent_chunk_id"),
                "parent_chunk_text": item.get("parent_chunk_text"),
                "parent_page_range": item.get("parent_page_range")
            }

        fused_results = []
        for cid in all_chunk_ids:
            vector_rank = vector_ranks.get(cid)
            bm25_rank = bm25_ranks.get(cid)

            vector_contrib = 1.0 / (K_RRF + vector_rank) if vector_rank else 0.0
            bm25_contrib = 1.0 / (K_RRF + bm25_rank) if bm25_rank else 0.0

            # Weighted RRF: vector contributes 0.7, BM25 contributes 0.3
            fused_score = 0.7 * vector_contrib + 0.3 * bm25_contrib

            meta = chunk_metadata[cid]
            fused_results.append({
                "id": cid,
                "chunk_text": meta["chunk_text"],
                "text": meta["chunk_text"], # compatibility fallback
                "page_number": meta["page_number"],
                "chunk_index": meta["chunk_index"],
                "fused_score": fused_score,
                "parent_chunk_id": meta.get("parent_chunk_id"),
                "parent_chunk_text": meta.get("parent_chunk_text"),
                "parent_page_range": meta.get("parent_page_range")
            })

        # Sort by fused score descending
        fused_results = sorted(fused_results, key=lambda x: x["fused_score"], reverse=True)
        return fused_results

    def delete_document_collection(self, doc_id: uuid.UUID) -> None:
        """
        Hard deletes the corresponding document collection from the vector database.
        """
        client = self._get_client()
        collection_name = self.get_collection_name(doc_id)
        logger.info(f"Deleting ChromaDB collection: '{collection_name}' for doc ID '{doc_id}'")
        try:
            client.delete_collection(name=collection_name)
        except Exception as e:
            logger.warning(f"ChromaDB collection deletion skipped or failed: {e}")

# Instantiate singleton vector store service instance
vector_store_service = VectorStoreService()
