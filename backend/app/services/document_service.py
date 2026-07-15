import uuid
import logging
import re
import os
from sqlalchemy.orm import Session
from fastapi import UploadFile
from app.models import User
from app.core.exceptions import InvalidFileTypeException, FileTooLargeException, CorruptedFileException
from app.services.pdf_parser_service import parse_pdf_pages
from app.services.chunker_service import chunk_document_pages
from app.services.embedding_service import embedding_service
from app.services.vector_store_service import vector_store_service
from app.repositories import create_document, soft_delete_document

logger = logging.getLogger("app.services.document")

# 10MB limit
MAX_FILE_SIZE = 10 * 1024 * 1024

def sanitize_filename(filename: str) -> str:
    """
    Sanitize the user-supplied filename.
    Strips path separators and replaces special characters with underscores.
    """
    # Remove directory path parts if present
    base = os.path.basename(filename)
    # Allow only letters, numbers, dot, dash, and underscore
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    return sanitized[:255]

class DocumentService:
    """
    Handles PDF validation, text parsing/extraction, and DB storage.
    """
    def process_pdf_upload(
        self,
        db: Session,
        file: UploadFile,
        content_length: int | None,
        current_user: User | None,
        session_id: str | None
    ) -> dict:
        # 1. Verify file size using HTTP Content-Length header if available
        if content_length is not None and content_length > MAX_FILE_SIZE:
            raise FileTooLargeException(
                f"File size exceeds maximum limit of 10MB (Header check: {content_length} bytes)."
            )

        # 2. Read file bytes and verify actual size in memory
        try:
            file_bytes = file.file.read()
        except Exception as read_err:
            raise CorruptedFileException(f"Failed to read uploaded file: {str(read_err)}")

        actual_size = len(file_bytes)
        if actual_size == 0:
            raise CorruptedFileException("Uploaded file is empty (0 bytes).")
        if actual_size > MAX_FILE_SIZE:
            raise FileTooLargeException(
                f"File size exceeds maximum limit of 10MB (Actual check: {actual_size} bytes)."
            )

        # 3. Verify magic bytes (%PDF- signature) to ensure it is a valid PDF
        if len(file_bytes) < 4 or file_bytes[:4] != b"%PDF":
            raise InvalidFileTypeException("Invalid file format signature. Only valid PDF files are supported.")

        # 4. Sanitize file name
        filename_raw = file.filename or "uploaded_document.pdf"
        sanitized_name = sanitize_filename(filename_raw)

        # 5. Extract text page-by-page (PyMuPDF + Tesseract fallback)
        try:
            pages_data, ocr_triggered = parse_pdf_pages(file_bytes)
        except Exception as parse_err:
            # Propagate custom exception classes raised by the parser
            if hasattr(parse_err, "status_code"):
                raise parse_err
            raise CorruptedFileException(f"Could not parse PDF document structure: {str(parse_err)}")

        # 6. Chunk text page-by-page
        chunks = chunk_document_pages(pages_data)

        # 8. Detect document language using langdetect (runs after parsing/OCR text is extracted)
        full_text = "\n".join([page["text"] for page in pages_data])
        detected_lang = None
        if full_text.strip():
            try:
                import langdetect
                detected_lang = langdetect.detect(full_text)
                logger.info(f"Detected document language: '{detected_lang}'")
            except Exception as lang_err:
                logger.warning(f"Language detection failed: {lang_err}. Defaulting to 'en'.")
                detected_lang = "en"
        else:
            detected_lang = "en"

        # 7. Generate embeddings (using primary model or fallback sentence-transformers)
        active_model = embedding_service.get_embedding_model_info(lang=detected_lang)
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedding_service.generate_embeddings(chunk_texts, lang=detected_lang)

        # NOTE: Duplicate upload check (equivalent content hashing) is currently skipped for MVP.
        # This is a known limitation. We do not check if a collection with equivalent file content hash exists,
        # and instead index every upload as a unique document with its own isolated collection.

        # 9. Write database row via repository layer (initially without chroma_collection_id)
        user_id = current_user.id if current_user else None
        # For authenticated users, audit log has session_id, but user_id is the primary ownership
        doc_session_id = session_id if current_user is None else None

        # OLD: created document without passing language detection parameter — replaced below to add detected_language
        # doc_record = create_document(
        #     db=db,
        #     filename=sanitized_name,
        #     page_count=len(pages_data),
        #     ocr_triggered=ocr_triggered,
        #     chroma_collection_id=None,
        #     user_id=user_id,
        #     session_id=doc_session_id,
        #     embedding_model=active_model
        # )

        doc_record = create_document(
            db=db,
            filename=sanitized_name,
            page_count=len(pages_data),
            ocr_triggered=ocr_triggered,
            chroma_collection_id=None,
            user_id=user_id,
            session_id=doc_session_id,
            embedding_model=active_model,
            detected_language=detected_lang
        )

        # 10. Generate deterministic collection name from created document ID
        chroma_collection_id = vector_store_service.get_collection_name(doc_record.id)
        doc_record.chroma_collection_id = chroma_collection_id
        db.commit()
        db.refresh(doc_record)

        # Save original PDF file to disk
        try:
            uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            file_path = os.path.join(uploads_dir, f"{doc_record.id}.pdf")
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            logger.info(f"Saved original PDF file to disk at: {file_path}")
        except Exception as save_err:
            logger.error(f"Failed to save original PDF file to disk: {save_err}")
            raise CorruptedFileException(f"Failed to save document file on server: {str(save_err)}")

        # 11. Index chunks and embeddings in ChromaDB
        vector_store_service.store_document_chunks(doc_record.id, chunks, embeddings)

        logger.info(f"Processed document upload successfully. Document ID: {doc_record.id}")

        # Return exact snake_case values as expected by the frontend's mock endpoints
        return {
            "document_id": doc_record.id,
            "filename": doc_record.filename,
            "page_count": doc_record.page_count,
            "ocr_triggered": doc_record.ocr_triggered
        }

    # OLD: soft deletes PostgreSQL document and hard deletes vector storage only — replaced below to also invalidate cached RAG responses
    # def delete_document(self, db: Session, doc_id: uuid.UUID) -> bool:
    #     """
    #     Soft deletes the document in PostgreSQL database and hard deletes
    #     the vector index collection in ChromaDB.
    #     """
    #     # 1. Soft delete database row
    #     success = soft_delete_document(db, doc_id)
    #     if not success:
    #         return False
    # 
    #     # 2. Hard delete vector collection in ChromaDB
    #     vector_store_service.delete_document_collection(doc_id)
    #     return True

    def delete_document(self, db: Session, doc_id: uuid.UUID) -> bool:
        """
        Soft deletes the document in PostgreSQL database, hard deletes the vector collection in ChromaDB,
        and invalidates cached RAG responses for this document.
        """
        # 1. Soft delete database row
        success = soft_delete_document(db, doc_id)
        if not success:
            return False

        # 2. Hard delete vector collection in ChromaDB
        vector_store_service.delete_document_collection(doc_id)

        # 3. Clear RAG response cache for this document
        try:
            from app.services.chat_service import rag_response_cache
            doc_id_str = str(doc_id)
            if doc_id_str in rag_response_cache:
                rag_response_cache.pop(doc_id_str, None)
                logger.info(f"Cleared RAG response cache for document '{doc_id}'")
        except Exception as cache_err:
            logger.warning(f"Could not clear RAG response cache for document '{doc_id}': {cache_err}")

        # 4. Delete the physical PDF file if it exists
        try:
            uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads")
            file_path = os.path.join(uploads_dir, f"{doc_id}.pdf")
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted physical PDF file for document '{doc_id}'")
        except Exception as file_err:
            logger.warning(f"Could not delete physical PDF file for document '{doc_id}': {file_err}")

        return True

# Instantiate singleton service instance
document_service = DocumentService()
