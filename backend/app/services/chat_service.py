import logging
import uuid
import json
import time
from datetime import datetime
from typing import Generator, Dict, Any, List
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.exceptions import (
    EmbeddingModelMismatchException,
    DocumentNotIndexedException,
    RateLimitException,
    ExternalServiceException,
    AppException,
    OllamaUnavailableException
)
from app.models.document import Document
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.monthly_usage import MonthlyUsage
from app.services.embedding_service import embedding_service
from app.services.vector_store_service import vector_store_service
from app.services.llm_provider import get_llm_provider

logger = logging.getLogger("app.services.chat")

# Module-level in-memory RAG response cache
# Structure: { doc_id_str: { cache_key_hash: {"answer": str, "citations": list} } }
rag_response_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

class ChatService:
    def get_or_create_monthly_usage(self, db: Session) -> MonthlyUsage:
        """
        Retrieves or creates the MonthlyUsage spending counter for the current month (YYYY-MM).
        """
        current_month = datetime.utcnow().strftime("%Y-%m")
        usage = db.query(MonthlyUsage).filter(MonthlyUsage.month == current_month).first()
        if not usage:
            try:
                usage = MonthlyUsage(
                    id=uuid.uuid4(),
                    month=current_month,
                    total_tokens=0,
                    total_cost_estimate=0.0
                )
                db.add(usage)
                db.commit()
                db.refresh(usage)
            except Exception:
                # Concurrent transaction handling
                db.rollback()
                usage = db.query(MonthlyUsage).filter(MonthlyUsage.month == current_month).first()
                if not usage:
                    raise ExternalServiceException("Failed to initialize monthly usage limits tracker.")
        return usage

    def generate_chat_stream(
        self,
        db: Session,
        document_id: uuid.UUID,
        conversation_id: uuid.UUID | None,
        question: str,
        request_id: uuid.UUID,
        user_id: uuid.UUID | None,
        session_id: str | None
    ) -> Generator[str, None, None]:
        """
        Executes the RAG pipeline and returns a chunked HTTP generator stream.
        """
        # 1. Idempotency Check: check if message with request_id already exists
        existing_msg = db.query(Message).filter(Message.request_id == str(request_id)).first()
        if existing_msg:
            logger.info(f"Idempotent match detected for request_id '{request_id}'. Re-streaming stored reply.")
            def stream_existing():
                content = existing_msg.content
                # Stream token-by-token (words) to mimic live streaming compatibility
                for word in content.split(" "):
                    yield json.dumps({"token": word + " "}) + "\n"
                
                citations = existing_msg.source_citations or []
                yield json.dumps({
                    "citations": citations,
                    "conversationId": str(existing_msg.conversation_id)
                }) + "\n"
            return stream_existing()

        # OLD: blocked RAG request if monthly limit reached for all providers — replaced below to bypass limits for free local Ollama calls
        # monthly_usage = self.get_or_create_monthly_usage(db)
        # monthly_usage_id = monthly_usage.id
        # if monthly_usage.total_cost_estimate >= settings.MONTHLY_SPEND_LIMIT_USD:
        #     logger.warning(f"Monthly spend limit reached ({monthly_usage.total_cost_estimate} >= {settings.MONTHLY_SPEND_LIMIT_USD})")
        #     raise RateLimitException("Monthly usage limit reached")

        monthly_usage = self.get_or_create_monthly_usage(db)
        monthly_usage_id = monthly_usage.id
        if settings.LLM_PROVIDER.lower() != "ollama" and monthly_usage.total_cost_estimate >= settings.MONTHLY_SPEND_LIMIT_USD:
            logger.warning(f"Monthly spend limit reached ({monthly_usage.total_cost_estimate} >= {settings.MONTHLY_SPEND_LIMIT_USD})")
            raise RateLimitException("Monthly usage limit reached")

        # 3. Fetch Document and check indexing model integrity
        doc = db.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted == False
        ).first()
        
        if not doc:
            raise DocumentNotIndexedException("Document does not exist or has been deleted.")

        active_model = embedding_service.get_embedding_model_info(lang=doc.detected_language)
        if doc.embedding_model != active_model:
            logger.error(f"Model mismatch! Document indexed with '{doc.embedding_model}', queried with '{active_model}'")
            raise EmbeddingModelMismatchException(
                f"Embedding model mismatch: Query model '{active_model}' does not match document index model '{doc.embedding_model}'."
            )

        # 4. Generate query embedding and fetch similar chunks
        try:
            query_emb = embedding_service.generate_query_embedding(question, lang=doc.detected_language)
        except Exception as embed_err:
            logger.error(f"RAG query embedding generation failed: {embed_err}")
            raise ExternalServiceException("Failed to process question text vector mapping.")

        # OLD: retrieved chunks from query_similar_chunks were passed directly into 
        # the prompt with no reranking step — replaced below to add a cross-encoder 
        # reranking pass on top of the hybrid-search candidates from Prompt I, 
        # improving final chunk selection quality before it reaches the LLM.
        # similar_chunks = vector_store_service.query_hybrid(
        #     query_text=question,
        #     query_embedding=query_emb,
        #     collection_id=doc.id,
        #     vector_top_k=15,
        #     bm25_top_k=15
        # )

        # 1. Fetch candidate chunks using hybrid search (top 15 vector, top 15 BM25)
        candidates = vector_store_service.query_hybrid(
            query_text=question,
            query_embedding=query_emb,
            collection_id=doc.id,
            vector_top_k=settings.VECTOR_TOP_K,
            bm25_top_k=settings.BM25_TOP_K
        )

        # 2. Rerank candidates using CrossEncoder (top 5 final)
        from app.services.reranker_service import reranker_service
        similar_chunks = reranker_service.rerank_chunks(
            question=question,
            candidate_chunks=candidates,
            top_n=settings.FINAL_TOP_N
        )

        # 5. Conversation Memory/Creation logic
        if conversation_id:
            # Validate ownership/existence
            conv = db.query(Conversation).filter(
                Conversation.id == conversation_id,
                Conversation.is_deleted == False
            ).first()
            if not conv:
                raise AppException("Conversation not found or has been soft deleted.", status_code=404)
        else:
            # Create a new conversation mapping title
            title = question[:37] + "..." if len(question) > 40 else question
            conv = Conversation(
                id=uuid.uuid4(),
                user_id=user_id,
                session_id=session_id if user_id is None else None,
                document_id=document_id,
                title=title
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)
            conversation_id = conv.id

        # Fetch last N messages memory
        history = db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.is_deleted == False
        ).order_by(Message.created_at.asc()).all()[-settings.CONVERSATION_MEMORY_LIMIT:]

        # OLD: stream_generator function without RAG response caching layer — replaced below to add MD5 hashing cache checks and response cache storage
        # def stream_generator():
        #     accumulated_text = ""
        #     citations = []
        #     
        #     # Format context from matching chunks
        #     context_blocks = []
        #     for chunk in similar_chunks:
        #         page_info = chunk.get("page_number", 0)
        #         snippet_text = chunk.get("text", "")
        #         # Save citations format
        #         citations.append({
        #             "page": page_info,
        #             "snippet": snippet_text
        #         })
        #         context_blocks.append(f"[Page {page_info}]: {snippet_text}")
        # 
        #     context_str = "\n\n".join(context_blocks)
        # 
        #     # Format memory logs
        #     memory_blocks = []
        #     for msg in history:
        #         memory_blocks.append(f"{msg.role.upper()}: {msg.content}")
        #     memory_str = "\n".join(memory_blocks)
        # 
        #     # System Prompt with explicit instructions
        #     system_prompt = (
        #         "You are a precise document analyst. Your persona is concise, objective, and truthful.\n"
        #         "You must answer the user's question ONLY using the context provided below. Citing page numbers is critical.\n"
        #         "If the context does not contain the answer, you must respond EXACTLY with: "
        #         "\"I could not find this information in the document\"\n"
        #         "Do NOT make up information or hallucinate facts beyond the given context block."
        #     )
        # 
        #     # Prompt template
        #     prompt = (
        #         f"DOCUMENT CONTEXT:\n{context_str}\n\n"
        #         f"CONVERSATION HISTORY:\n{memory_str}\n\n"
        #         f"USER QUESTION: {question}\n\n"
        #         f"YOUR CITATION-GROUNDED ANSWER:"
        #     )
        # 
        #     # Call provider
        #     try:
        #         llm = get_llm_provider()
        #         stream = llm.generate_stream(
        #             prompt=prompt,
        #             system_prompt=system_prompt,
        #             temperature=settings.LLM_TEMPERATURE,
        #             max_tokens=settings.LLM_MAX_TOKENS
        #         )
        # 
        #         # Track usage tokens returned at the end
        #         token_usage = None
        # 
        #         for chunk in stream:
        #             if chunk["type"] == "token":
        #                 token_val = chunk["token"]
        #                 accumulated_text += token_val
        #                 # Send JSON chunk token to match frontend Reader
        #                 yield json.dumps({"token": token_val}) + "\n"
        #             elif chunk["type"] == "usage":
        #                 token_usage = chunk
        # 
        #         # Verify answer contents against hallucination trigger
        #         is_grounded_answer = "i could not find this information in the document" not in accumulated_text.lower().strip()
        #         final_citations = citations if is_grounded_answer else []
        # 
        #         # Final metadata packet containing citations
        #         yield json.dumps({
        #             "citations": final_citations,
        #             "conversationId": str(conversation_id)
        #         }) + "\n"
        # 
        #         # 7. Persist messages AND MonthlyUsage updates upon successful stream completion
        #         # Ensure we reload session bounds to commit cleanly in generator
        #         try:
        #             db.expire_all()
        #             
        #             # Store user question message
        #             user_msg = Message(
        #                 id=uuid.uuid4(),
        #                 conversation_id=conversation_id,
        #                 role="user",
        #                 content=question,
        #                 source_citations=None,
        #                 request_id=None
        #             )
        #             db.add(user_msg)
        # 
        #             # Store assistant cited answer message
        #             assistant_msg = Message(
        #                 id=uuid.uuid4(),
        #                 conversation_id=conversation_id,
        #                 role="assistant",
        #                 content=accumulated_text,
        #                 source_citations=final_citations,
        #                 request_id=str(request_id)
        #             )
        #             db.add(assistant_msg)
        # 
        #             # Increment usage limits
        #             if token_usage:
        #                 # Re-fetch monthly usage record to lock it
        #                 usage_rec = db.query(MonthlyUsage).filter(MonthlyUsage.id == monthly_usage_id).first()
        #                 if usage_rec:
        #                     usage_rec.total_tokens += (token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0))
        #                     usage_rec.total_cost_estimate += token_usage.get("cost", 0.0)
        #             
        #             db.commit()
        #             logger.info(f"Successfully persisted conversation messages for request '{request_id}'")
        # 
        #         except Exception as save_err:
        #             db.rollback()
        #             logger.error(f"Failed to save RAG conversation memory to database: {save_err}")
        # 
        #     except Exception as stream_err:
        #         # Catch mid-stream failures and emit standard error bubble chunk to frontend
        #         logger.error(f"Error occurred during LLM stream: {stream_err}")
        #         yield json.dumps({"error": f"Model stream execution failed: {str(stream_err)}"}) + "\n"

        # Format memory logs outside the generator thread to prevent DetachedInstanceError
        memory_blocks = []
        for msg in history:
            memory_blocks.append(f"{msg.role.upper()}: {msg.content}")
        memory_str = "\n".join(memory_blocks)

        # Extract document embedding model outside the generator thread
        doc_embedding_model = doc.embedding_model

        def stream_generator():
            from app.core.database import SessionLocal
            
            accumulated_text = ""
            citations = []
            
            # OLD: after reranking, the child chunk's own (small) text was sent 
            # directly into the <context> XML block — replaced below to swap in each 
            # matched child chunk's PARENT chunk text instead, giving the LLM fuller 
            # surrounding context while retrieval matching still happened at the 
            # precise child-chunk level.
            # similar_chunks = similar_chunks  [old version, kept for reference]

            seen_parents = set()
            unique_parents = []
            for chunk in similar_chunks:
                p_id = chunk.get("parent_chunk_id")
                # Fallback to chunk ID if parent_chunk_id is missing (e.g. from legacy tests/mocks)
                p_key = p_id if p_id else chunk.get("id")
                if p_key not in seen_parents:
                    seen_parents.add(p_key)
                    unique_parents.append(chunk)

            # Format context from matching parent chunks
            context_blocks = []
            for chunk in unique_parents:
                p_text = chunk.get("parent_chunk_text") or chunk.get("text", "") or chunk.get("chunk_text", "")
                p_page = chunk.get("parent_page_range") or str(chunk.get("page_number", 0))
                # Save citations format
                citations.append({
                    "page": p_page,
                    "snippet": p_text
                })
                # XML style tag formatting per chunk
                context_blocks.append(f'  <chunk page="{p_page}">{p_text}</chunk>')

            context_str = "\n".join(context_blocks)
            context_xml = f"<context>\n{context_str}\n</context>"
            history_xml = f"<conversation_history>\n{memory_str}\n</conversation_history>"
            question_xml = f"<question>{question}</question>"

            # Generate normalized cache key
            doc_id_str = str(document_id)
            normalized_q = question.strip().lower()
            chunk_ids = sorted([str(c.get("id")) for c in similar_chunks])
            chunk_ids_str = ",".join(chunk_ids)
            
            import hashlib
            raw_key_string = f"{doc_id_str}:{doc_embedding_model}:{normalized_q}:{chunk_ids_str}"
            cache_key = hashlib.md5(raw_key_string.encode("utf-8")).hexdigest()
            
            # Check RAG response cache (hit check)
            cached_data = rag_response_cache.get(doc_id_str, {}).get(cache_key)
            
            # Create a fresh database session for the background streaming generator thread
            gen_db = SessionLocal()
            try:
                # 3. Check rerank confidence threshold on top candidate
                top_chunk_score = similar_chunks[0].get("rerank_score", -1.0) if similar_chunks else -1.0
                if top_chunk_score < settings.MIN_RERANK_CONFIDENCE:
                    logger.info(
                        f"RERANK CONFIDENCE THRESHOLD TRIPPED: Top chunk score ({top_chunk_score:.4f}) is below "
                        f"MIN_RERANK_CONFIDENCE ({settings.MIN_RERANK_CONFIDENCE}). "
                        f"Skipping LLM call for document '{document_id}' and question '{question}'."
                    )
                    
                    fallback_response = "I could not find relevant information in the document to answer this question."
                    
                    # Stream fallback response back token-by-token
                    for word in fallback_response.split(" "):
                        if word:
                            yield json.dumps({"token": word + " "}) + "\n"
                            
                    yield json.dumps({
                        "citations": [],
                        "conversationId": str(conversation_id)
                    }) + "\n"
                    
                    # Persist conversation message history
                    try:
                        user_msg = Message(
                            id=uuid.uuid4(),
                            conversation_id=conversation_id,
                            role="user",
                            content=question,
                            source_citations=None,
                            request_id=None
                        )
                        gen_db.add(user_msg)
                        assistant_msg = Message(
                            id=uuid.uuid4(),
                            conversation_id=conversation_id,
                            role="assistant",
                            content=fallback_response,
                            source_citations=[],
                            request_id=str(request_id)
                        )
                        gen_db.add(assistant_msg)
                        gen_db.commit()
                        logger.info(f"Successfully persisted fallback response messages for request '{request_id}'")
                    except Exception as save_err:
                        gen_db.rollback()
                        logger.error(f"Failed to save fallback response messages to database: {save_err}")
                    return

                if cached_data:
                    logger.info(f"RAG CACHE HIT: Found cached response for document '{document_id}' and question '{question}'")
                    cached_answer = cached_data["answer"]
                    cached_citations = cached_data["citations"]
                    
                    # Stream cached answer back token-by-token
                    for word in cached_answer.split(" "):
                        if word:
                            yield json.dumps({"token": word + " "}) + "\n"
                    
                    yield json.dumps({
                        "citations": cached_citations,
                        "conversationId": str(conversation_id)
                    }) + "\n"
                    
                    # Persist messages to DB so history remains intact
                    try:
                        user_msg = Message(
                            id=uuid.uuid4(),
                            conversation_id=conversation_id,
                            role="user",
                            content=question,
                            source_citations=None,
                            request_id=None
                        )
                        gen_db.add(user_msg)
                        assistant_msg = Message(
                            id=uuid.uuid4(),
                            conversation_id=conversation_id,
                            role="assistant",
                            content=cached_answer,
                            source_citations=cached_citations,
                            request_id=str(request_id)
                        )
                        gen_db.add(assistant_msg)
                        gen_db.commit()
                        logger.info(f"Successfully persisted cached conversation messages for request '{request_id}'")
                    except Exception as save_err:
                        gen_db.rollback()
                        logger.error(f"Failed to save cached RAG conversation memory to database: {save_err}")
                    return

                logger.info(f"RAG CACHE MISS: No cached response for document '{document_id}' and question '{question}'")

                # OLD: old system prompt for QA grounding
                # system_prompt = (
                #     "You are a precise document analyst. Your persona is concise, objective, and truthful.\n"
                #     "You must answer the user's question ONLY using the context provided below. Citing page numbers is critical.\n"
                #     "If the context does not contain the answer, you must respond EXACTLY with: "
                #     "\"I could not find this information in the document\"\n"
                #     "Do NOT make up information or hallucinate facts beyond the given context block."
                # )
                system_prompt = (
                    "You are a precise document analyst. Be concise, objective, and truthful.\n"
                    "Answer ONLY using information found inside the <context> tags above.\n"
                    "If the answer is not present in the context, respond exactly with: "
                    "'I could not find this information in the document'\n"
                    "Always cite the page number(s) you used, in the format (Page X), "
                    "referencing the page attribute from the chunk(s) you drew from."
                )

                # OLD: injected retrieved chunks as plain concatenated text 
                # ("Based on the following context: {chunks}...") with no structural 
                # separation between chunks or their page numbers — replaced below with 
                # XML-tagged structure for clearer grounding and more reliable citation 
                # extraction by the LLM.
                # prompt = f"Based on the following context: {chunks}\n\nAnswer: {question}"

                # Prompt template built using XML tags
                prompt = (
                    f"{context_xml}\n\n"
                    f"{history_xml}\n\n"
                    f"{question_xml}"
                )

                # Call provider
                try:
                    # OLD: always called the LLM with whatever chunks were retrieved, even if 
                    # none were meaningfully relevant to the question — replaced below with a 
                    # confidence threshold check that skips the LLM call entirely when 
                    # retrieval quality is too low, returning a direct "not found" response 
                    # instead.
                    # stream = llm.generate_stream(prompt)   [old unconditional call]
                    llm = get_llm_provider()
                    stream = llm.generate_stream(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=0.0,
                        max_tokens=settings.LLM_MAX_TOKENS
                    )

                    # Track usage tokens returned at the end
                    token_usage = None

                    for chunk in stream:
                        if chunk["type"] == "token":
                            token_val = chunk["token"]
                            accumulated_text += token_val
                            # Send JSON chunk token to match frontend Reader
                            yield json.dumps({"token": token_val}) + "\n"
                        elif chunk["type"] == "usage":
                            token_usage = chunk

                    # Verify answer contents against hallucination trigger
                    is_grounded_answer = "i could not find this information in the document" not in accumulated_text.lower().strip()
                    final_citations = citations if is_grounded_answer else []

                    # Final metadata packet containing citations
                    yield json.dumps({
                        "citations": final_citations,
                        "conversationId": str(conversation_id)
                    }) + "\n"

                    # 7. Persist messages AND MonthlyUsage updates upon successful stream completion
                    try:
                        # Store user question message
                        user_msg = Message(
                            id=uuid.uuid4(),
                            conversation_id=conversation_id,
                            role="user",
                            content=question,
                            source_citations=None,
                            request_id=None
                        )
                        gen_db.add(user_msg)

                        # Store assistant cited answer message
                        assistant_msg = Message(
                            id=uuid.uuid4(),
                            conversation_id=conversation_id,
                            role="assistant",
                            content=accumulated_text,
                            source_citations=final_citations,
                            request_id=str(request_id)
                        )
                        gen_db.add(assistant_msg)

                        # Increment usage limits
                        if token_usage:
                            # Re-fetch monthly usage record to lock it
                            usage_rec = gen_db.query(MonthlyUsage).filter(MonthlyUsage.id == monthly_usage_id).first()
                            if usage_rec:
                                usage_rec.total_tokens += (token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0))
                                usage_rec.total_cost_estimate += token_usage.get("cost", 0.0)
                        
                        gen_db.commit()
                        logger.info(f"Successfully persisted conversation messages for request '{request_id}'")

                        # Store result in RAG response cache
                        if doc_id_str not in rag_response_cache:
                            rag_response_cache[doc_id_str] = {}
                        rag_response_cache[doc_id_str][cache_key] = {
                            "answer": accumulated_text,
                            "citations": final_citations
                        }
                        logger.info(f"RAG CACHE STORE: Stored answer in cache for key '{cache_key}'")

                    except Exception as save_err:
                        gen_db.rollback()
                        logger.error(f"Failed to save RAG conversation memory to database: {save_err}")

                except Exception as stream_err:
                    # Catch mid-stream failures and emit standard error bubble chunk to frontend
                    logger.error(f"Error occurred during LLM stream: {stream_err}")
                    error_msg = str(stream_err)
                    if not isinstance(stream_err, OllamaUnavailableException):
                        error_msg = f"Model stream execution failed: {error_msg}"
                    yield json.dumps({"error": error_msg}) + "\n"
            finally:
                gen_db.close()

        return stream_generator()

# Instantiate singleton service instance
chat_service = ChatService()
