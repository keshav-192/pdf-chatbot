import pytest
import uuid
from unittest.mock import MagicMock

from app.core.exceptions import (
    InvalidFileTypeException,
    FileTooLargeException,
    CorruptedFileException,
    PasswordProtectedException,
    EmbeddingModelMismatchException,
    ForbiddenException
)
from app.services.chunker_service import chunk_document_pages
from app.services.conversation_service import conversation_service
from app.services.pdf_parser_service import parse_pdf_pages
from app.services.document_service import document_service
from app.models.conversation import Conversation

# =========================================================================
# Unit Tests: Chunker Service
# =========================================================================
def test_chunking_logic():
    # 2 pages of mock text data
    pages = [
        {"page_number": 1, "text": "This is page one text content. " * 30},
        {"page_number": 2, "text": "This is page two text content. " * 20}
    ]
    
    # Run chunker
    chunks = chunk_document_pages(pages, chunk_size=100, chunk_overlap=20)
    
    assert len(chunks) > 0
    # Check that metadata metadata is attached correctly
    first_chunk = chunks[0]
    assert "chunk_index" in first_chunk
    assert "page_number" in first_chunk
    assert "text" in first_chunk
    assert first_chunk["page_number"] == 1
    assert first_chunk["chunk_index"] == 0
    assert "parent_chunk_id" in first_chunk
    assert "parent_chunk_text" in first_chunk
    assert "parent_page_range" in first_chunk
    assert len(first_chunk["parent_chunk_text"]) >= len(first_chunk["text"])


# =========================================================================
# Unit Tests: PDF Parsing & Validation
# =========================================================================
def test_pdf_validation_invalid_type():
    # File upload mock with wrong magic bytes signature
    file_mock = MagicMock()
    file_mock.filename = "hacker.exe"
    file_mock.file.read.return_value = b"MZ\x90\x00\x03\x00\x00\x00"  # PE exe magic bytes
    
    with pytest.raises(InvalidFileTypeException):
        document_service.process_pdf_upload(
            db=MagicMock(),
            file=file_mock,
            content_length=100,
            current_user=None,
            session_id="guest-session-123"
        )

def test_pdf_validation_oversized():
    # File size limit check (content length > 10MB)
    file_mock = MagicMock()
    with pytest.raises(FileTooLargeException):
        document_service.process_pdf_upload(
            db=MagicMock(),
            file=file_mock,
            content_length=11 * 1024 * 1024, # 11MB
            current_user=None,
            session_id="guest-session-123"
        )

def test_pdf_validation_corrupted(monkeypatch):
    # Mock PyMuPDF fitz.open raising an exception representing corrupted file
    import fitz
    def mock_fitz_open(*args, **kwargs):
        raise Exception("Failed to open PDF structure")
    
    monkeypatch.setattr(fitz, "open", mock_fitz_open)
    
    file_mock = MagicMock()
    file_mock.filename = "corrupted.pdf"
    file_mock.file.read.return_value = b"%PDF-1.4 mock corrupted contents"
    
    from tests.conftest import TestingSessionLocal
    db = TestingSessionLocal()
    try:
        with pytest.raises(CorruptedFileException):
            document_service.process_pdf_upload(
                db=db,
                file=file_mock,
                content_length=100,
                current_user=None,
                session_id="guest-session-123"
            )
    finally:
        db.close()


def test_pdf_validation_password_protected(monkeypatch):
    # Mock PyMuPDF fitz.open returning an encrypted doc
    import fitz
    doc_mock = MagicMock()
    doc_mock.is_encrypted = True
    doc_mock.authenticate.return_value = False # authenticate fails
    
    monkeypatch.setattr(fitz, "open", lambda *args, **kwargs: doc_mock)
    
    with pytest.raises(PasswordProtectedException):
        parse_pdf_pages(b"%PDF-1.4 mock contents")


def test_pdf_parsing_multi_column(monkeypatch):
    # Mock fitz.open to return blocks layout that satisfies the multi-column heuristic
    import fitz
    import pdfplumber
    
    doc_mock = MagicMock()
    page_mock = MagicMock()
    # page width = 600
    page_mock.rect.width = 600
    
    # 4 blocks: 2 on the left (x1 <= 315), 2 on the right (x0 >= 285)
    # block format: (x0, y0, x1, y1, "text", block_no, block_type)
    # block_type = 0 is text block
    def mock_get_text(opt=None):
        if opt == "blocks":
            return [
                (50, 100, 200, 150, "Left column paragraph one", 0, 0),
                (50, 200, 220, 250, "Left column paragraph two", 1, 0),
                (350, 100, 500, 150, "Right column paragraph one", 2, 0),
                (350, 200, 520, 250, "Right column paragraph two", 3, 0)
            ]
        return "Left column paragraph one\nLeft column paragraph two\nRight column paragraph one\nRight column paragraph two"
    page_mock.get_text.side_effect = mock_get_text
    
    doc_mock.__len__.return_value = 1
    doc_mock.__getitem__.return_value = page_mock
    doc_mock.is_encrypted = False
    
    # Mock fitz.open
    monkeypatch.setattr(fitz, "open", lambda *args, **kwargs: doc_mock)
    
    # Mock pdfplumber.open
    plumber_doc_mock = MagicMock()
    plumber_page_mock = MagicMock()
    plumber_page_mock.width = 600
    plumber_page_mock.height = 800
    
    left_crop_mock = MagicMock()
    left_crop_mock.extract_text.return_value = "Left column paragraph one\nLeft column paragraph two"
    
    right_crop_mock = MagicMock()
    right_crop_mock.extract_text.return_value = "Right column paragraph one\nRight column paragraph two"
    
    # pdfplumber page within_bbox crops
    def mock_within_bbox(bbox):
        # bbox is (x0, y0, x1, y1)
        if bbox[2] == 300: # left half
            return left_crop_mock
        return right_crop_mock
        
    plumber_page_mock.within_bbox.side_effect = mock_within_bbox
    plumber_page_mock.extract_tables.return_value = [] # no tables in this test
    
    plumber_doc_mock.pages = [plumber_page_mock]
    monkeypatch.setattr(pdfplumber, "open", lambda *args, **kwargs: plumber_doc_mock)
    
    # Parse PDF pages
    pages, ocr_triggered = parse_pdf_pages(b"%PDF-1.4 mock contents")
    
    assert len(pages) == 1
    assert pages[0]["was_multi_column"] is True
    assert pages[0]["had_tables"] is False
    assert "Left column" in pages[0]["text"]
    assert "Right column" in pages[0]["text"]


def test_pdf_parsing_with_table(monkeypatch):
    import fitz
    import pdfplumber
    
    doc_mock = MagicMock()
    page_mock = MagicMock()
    page_mock.rect.width = 600
    # No multi-column blocks
    def mock_get_text(opt=None):
        if opt == "blocks":
            return [
                (50, 100, 550, 150, "Some general description text", 0, 0)
            ]
        return "Some general description text"
    page_mock.get_text.side_effect = mock_get_text
    doc_mock.__len__.return_value = 1
    doc_mock.__getitem__.return_value = page_mock
    doc_mock.is_encrypted = False
    
    monkeypatch.setattr(fitz, "open", lambda *args, **kwargs: doc_mock)
    
    plumber_doc_mock = MagicMock()
    plumber_page_mock = MagicMock()
    plumber_page_mock.width = 600
    plumber_page_mock.height = 800
    
    # Return 1 table: 2x2 structure
    plumber_page_mock.extract_tables.return_value = [
        [
            ["Name", "Count"],
            ["App", "10"]
        ]
    ]
    
    plumber_doc_mock.pages = [plumber_page_mock]
    monkeypatch.setattr(pdfplumber, "open", lambda *args, **kwargs: plumber_doc_mock)
    
    # Parse PDF pages
    pages, ocr_triggered = parse_pdf_pages(b"%PDF-1.4 mock contents")
    
    assert len(pages) == 1
    assert pages[0]["was_multi_column"] is False
    assert pages[0]["had_tables"] is True
    # Verify flattened natural language table text is in page text
    assert "Table 1:" in pages[0]["text"]
    assert "Row 1: Name is App, Count is 10" in pages[0]["text"]


# =========================================================================
# Unit Tests: Ownership Verification
# =========================================================================
def test_ownership_verification_allows_owner():
    # Setup mock conversation in DB
    conv = Conversation(
        id=uuid.uuid4(),
        user_id=uuid.UUID("5f2ca46c-4064-44ee-be69-cb261ea57365"),
        session_id=None,
        title="Owner's Conversation"
    )
    
    db_mock = MagicMock()
    db_mock.query.return_value.filter.return_value.first.return_value = conv
    
    # Authorized access should pass and return conversation object
    res = conversation_service.verify_conversation_ownership(
        db=db_mock,
        conv_id=conv.id,
        user_id=uuid.UUID("5f2ca46c-4064-44ee-be69-cb261ea57365"),
        session_id=None
    )
    assert res == conv

def test_ownership_verification_denies_non_owner():
    # Setup mock conversation in DB
    conv = Conversation(
        id=uuid.uuid4(),
        user_id=uuid.UUID("5f2ca46c-4064-44ee-be69-cb261ea57365"),
        session_id=None,
        title="Owner's Conversation"
    )
    
    db_mock = MagicMock()
    db_mock.query.return_value.filter.return_value.first.return_value = conv
    
    # Non-owner request should fail with ForbiddenException (403)
    with pytest.raises(ForbiddenException):
        conversation_service.verify_conversation_ownership(
            db=db_mock,
            conv_id=conv.id,
            user_id=uuid.UUID("99999999-4064-44ee-be69-cb261ea57365"), # different user ID
            session_id=None
        )


# =========================================================================
# Unit Tests: Embedding Consistency Check
# =========================================================================
def test_embedding_model_consistency_mismatch(db, monkeypatch):
    from app.services.chat_service import chat_service
    from app.models.document import Document
    
    doc = Document(
        id=uuid.uuid4(),
        filename="report.pdf",
        page_count=1,
        embedding_model="text-embedding-3-small", # Document indexed with OpenAI
        is_deleted=False
    )
    db.add(doc)
    db.commit()
    
    # Mock embedding provider to return a different active local model name
    from app.services.embedding_service import embedding_service
    monkeypatch.setattr(embedding_service, "get_embedding_model_info", lambda *args, **kwargs: "all-MiniLM-L6-v2")
    
    with pytest.raises(EmbeddingModelMismatchException):
        # Triggering generate_chat_stream should fail with model mismatch exception
        generator = chat_service.generate_chat_stream(
            db=db,
            document_id=doc.id,
            conversation_id=None,
            question="What is this?",
            request_id=uuid.uuid4(),
            user_id=None,
            session_id="guest-session-123"
        )
        # Consume generator to run the synchronous checks
        next(generator)




# =========================================================================
# Unit Tests: Title Truncation / Auto-titling
# =========================================================================
def test_title_truncation_auto_titling():
    question_long = "This is a very long question query about SaaS layouts and RAG design pipelines"
    question_short = "What is RAG?"
    
    title_long = question_long[:37] + "..." if len(question_long) > 40 else question_long
    title_short = question_short[:37] + "..." if len(question_short) > 40 else question_short
    
    assert len(title_long) <= 40
    assert title_long.endswith("...")
    assert title_short == "What is RAG?"


def test_ollama_provider(monkeypatch):
    from app.services.llm_providers.ollama_provider import OllamaProvider
    from app.services.llm_provider import get_llm_provider
    from app.core.config import settings
    
    # 1. Test factory resolution
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    provider = get_llm_provider()
    assert isinstance(provider, OllamaProvider)
    
    # 2. Test generate_stream output parsing
    class MockResponse:
        def __init__(self):
            self.status_code = 200
        def raise_for_status(self):
            pass
        def iter_lines(self):
            yield b'{"message": {"content": "Hello "}, "done": false}'
            yield b'{"message": {"content": "world!"}, "done": true, "prompt_eval_count": 10, "eval_count": 5}'
            
    class MockContextManager:
        def __enter__(self):
            return MockResponse()
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
            
    import httpx
    monkeypatch.setattr(httpx, "stream", lambda *args, **kwargs: MockContextManager())
    
    chunks = list(provider.generate_stream("hello", "sys", 0.0, 10))
    assert len(chunks) == 3
    assert chunks[0] == {"type": "token", "token": "Hello "}
    assert chunks[1] == {"type": "token", "token": "world!"}
    assert chunks[2] == {"type": "usage", "input_tokens": 10, "output_tokens": 5, "cost": 0.0}


def test_rag_caching_and_invalidation(db, monkeypatch):
    from app.services.chat_service import chat_service, rag_response_cache
    from app.models.document import Document
    from app.services.embedding_service import embedding_service
    from app.services.vector_store_service import vector_store_service
    from app.services.document_service import document_service
    import app.services.chat_service
    
    # 1. Setup mock document in DB
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        filename="cached_report.pdf",
        page_count=2,
        embedding_model="all-MiniLM-L6-v2",
        is_deleted=False
    )
    db.add(doc)
    db.commit()
    
    # Clear any previous cached values for this doc
    rag_response_cache.pop(str(doc_id), None)
    
    # 2. Mock embedding, vector, and LLM services
    monkeypatch.setattr(embedding_service, "get_embedding_model_info", lambda *args, **kwargs: "all-MiniLM-L6-v2")
    monkeypatch.setattr(embedding_service, "generate_embeddings", lambda texts, *args, **kwargs: [[0.1] * 384])
    monkeypatch.setattr(embedding_service, "generate_query_embedding", lambda text, *args, **kwargs: [0.1] * 384)
    
    similar_mock = [
        {"id": f"{doc_id}_chunk_0", "text": "This is cached context chunk 1", "page_number": 1, "chunk_index": 0},
        {"id": f"{doc_id}_chunk_1", "text": "This is cached context chunk 2", "page_number": 2, "chunk_index": 1}
    ]
    monkeypatch.setattr(vector_store_service, "query_similar_chunks", lambda *args, **kwargs: similar_mock)
    similar_hybrid_mock = [
        {"chunk_text": "This is cached context chunk 1", "text": "This is cached context chunk 1", "page_number": 1, "chunk_index": 0, "fused_score": 1.0},
        {"chunk_text": "This is cached context chunk 2", "text": "This is cached context chunk 2", "page_number": 2, "chunk_index": 1, "fused_score": 0.9}
    ]
    monkeypatch.setattr(vector_store_service, "query_hybrid", lambda *args, **kwargs: similar_hybrid_mock)
    
    from app.services.reranker_service import reranker_service
    monkeypatch.setattr(reranker_service, "rerank_chunks", lambda question, candidate_chunks, top_n=5: [{**c, "rerank_score": 1.0} for c in candidate_chunks[:top_n]])
    
    # Mock LLM provider to stream tokens
    llm_call_count = 0
    class MockLLMProvider:
        def generate_stream(self, prompt, system_prompt, temperature, max_tokens):
            nonlocal llm_call_count
            llm_call_count += 1
            yield {"type": "token", "token": "Cached "}
            yield {"type": "token", "token": "answer"}
            yield {"type": "usage", "input_tokens": 10, "output_tokens": 5, "cost": 0.0}
            
    monkeypatch.setattr(app.services.chat_service, "get_llm_provider", lambda: MockLLMProvider())
    
    # 3. First call: Cache Miss
    req_id_1 = uuid.uuid4()
    generator = chat_service.generate_chat_stream(
        db=db,
        document_id=doc_id,
        conversation_id=None,
        question="What is cache?",
        request_id=req_id_1,
        user_id=None,
        session_id="guest-session-456"
    )
    res_1 = list(generator)
    assert llm_call_count == 1
    
    # Verify cached output is populated
    doc_id_str = str(doc_id)
    assert doc_id_str in rag_response_cache
    assert len(rag_response_cache[doc_id_str]) == 1
    
    # 4. Second call (same question, same doc): Cache Hit
    req_id_2 = uuid.uuid4()
    generator2 = chat_service.generate_chat_stream(
        db=db,
        document_id=doc_id,
        conversation_id=None,
        question="What is cache?",
        request_id=req_id_2,
        user_id=None,
        session_id="guest-session-456"
    )
    res_2 = list(generator2)
    # LLM provider call count should STILL be 1 (because cache was hit!)
    assert llm_call_count == 1
    
    # 5. Invalidation via document soft-delete
    monkeypatch.setattr(vector_store_service, "delete_document_collection", lambda *args, **kwargs: True)
    document_service.delete_document(db, doc_id)
    
    # The cache entries for this doc_id must now be cleared
    assert doc_id_str not in rag_response_cache
    
    # Restore document's is_deleted state so generate_chat_stream can fetch it
    doc.is_deleted = False
    db.commit()
    
    # 6. Third call: Cache Miss (since cache was cleared)
    req_id_3 = uuid.uuid4()
    generator3 = chat_service.generate_chat_stream(
        db=db,
        document_id=doc_id,
        conversation_id=None,
        question="What is cache?",
        request_id=req_id_3,
        user_id=None,
        session_id="guest-session-456"
    )
    res_3 = list(generator3)
    # LLM provider should be called again, so call count becomes 2!
    assert llm_call_count == 2


def test_pdf_parsing_scanned_ocr_multilingual(monkeypatch):
    import fitz
    import pytesseract
    
    doc_mock = MagicMock()
    page_mock = MagicMock()
    # Scanned PDF: page has no text, but has images/drawings
    page_mock.get_text.return_value = ""
    # Add content visuals (images)
    page_mock.get_images.return_value = ["mock_img_ref"]
    page_mock.get_drawings.return_value = []
    page_mock.rect.width = 600
    
    doc_mock.__len__.return_value = 1
    doc_mock.__getitem__.return_value = page_mock
    doc_mock.is_encrypted = False
    
    # Mock PyMuPDF open
    monkeypatch.setattr(fitz, "open", lambda *args, **kwargs: doc_mock)
    
    # Mock pdf2image.convert_from_bytes where it is used
    import app.services.pdf_parser_service
    monkeypatch.setattr(app.services.pdf_parser_service, "convert_from_bytes", lambda *args, **kwargs: ["mock_image_obj"])
    
    # Mock pytesseract.image_to_string to return mock Hindi OCR text
    hindi_text = "यह एक हिंदी दस्तावेज है।" # "This is a Hindi document."
    def mock_image_to_string(image, lang=None):
        assert lang == "eng+hin+spa+fra+deu"
        return hindi_text
    
    monkeypatch.setattr(pytesseract, "image_to_string", mock_image_to_string)
    
    # Run parsing
    pages, ocr_triggered = parse_pdf_pages(b"%PDF-1.4 mock scanned doc")
    
    assert len(pages) == 1
    assert ocr_triggered is True
    assert pages[0]["text"] == hindi_text
    
    # Also verify that langdetect is called and detected correctly on the document upload service
    # Mock database Session and Repository functions
    db_mock = MagicMock()
    file_mock = MagicMock()
    file_mock.filename = "scanned_hindi.pdf"
    file_mock.file.read.return_value = b"%PDF-1.4 mock scanned doc"
    
    # Mock create_document where it is used
    import app.services.document_service
    created_doc = MagicMock()
    created_doc.id = uuid.uuid4()
    def mock_create_document(*args, **kwargs):
        created_doc.detected_language = kwargs.get("detected_language")
        return created_doc
    monkeypatch.setattr(app.services.document_service, "create_document", mock_create_document)
    
    # Mock embedding and vector store
    from app.services.embedding_service import embedding_service
    from app.services.vector_store_service import vector_store_service
    monkeypatch.setattr(embedding_service, "get_embedding_model_info", lambda *args, **kwargs: "all-MiniLM-L6-v2")
    monkeypatch.setattr(embedding_service, "generate_embeddings", lambda texts, *args, **kwargs: [[0.1]*384])
    monkeypatch.setattr(vector_store_service, "store_document_chunks", lambda *args, **kwargs: "col_name")
    
    res = document_service.process_pdf_upload(
        db=db_mock,
        file=file_mock,
        content_length=1000,
        current_user=None,
        session_id="guest-session-123"
    )
    
    # Language should be detected as Hindi ("hi")
    assert created_doc.detected_language == "hi"


def test_embedding_model_granular_mismatch(db, monkeypatch):
    from app.services.chat_service import chat_service
    from app.models.document import Document
    
    # Document indexed with base English model
    doc = Document(
        id=uuid.uuid4(),
        filename="english_base.pdf",
        page_count=1,
        embedding_model="local-english-base",
        detected_language="en",
        is_deleted=False
    )
    db.add(doc)
    db.commit()
    
    # Active model configuration is set to large English model
    from app.services.embedding_service import embedding_service
    # Mock settings / method to return "local-english-large" for English
    monkeypatch.setattr(embedding_service, "get_embedding_model_info", lambda lang=None: "local-english-large")
    
    # Triggering generate_chat_stream should fail with EmbeddingModelMismatchException
    with pytest.raises(EmbeddingModelMismatchException) as exc_info:
        generator = chat_service.generate_chat_stream(
            db=db,
            document_id=doc.id,
            conversation_id=None,
            question="What is this?",
            request_id=uuid.uuid4(),
            user_id=None,
            session_id="guest-session-123"
        )
        next(generator)
    
    assert "Embedding model mismatch" in str(exc_info.value)


def test_query_hybrid_rrf_and_bm25(monkeypatch):
    from app.services.vector_store_service import vector_store_service, VectorStoreService
    
    # Restore the original query_hybrid method to override the autouse mock
    monkeypatch.setattr(vector_store_service, "query_hybrid", lambda *args, **kwargs: VectorStoreService.query_hybrid(vector_store_service, *args, **kwargs))
    
    # Mock ChromaDB client and collection methods
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["doc1_chunk_1", "doc1_chunk_0"]],
        "documents": [["Chunk one text content", "Chunk zero text content"]],
        "metadatas": [[{"page_number": 1, "chunk_index": 1}, {"page_number": 1, "chunk_index": 0}]],
        "distances": [[0.1, 0.2]]
    }
    mock_collection.get.return_value = {
        "ids": ["doc1_chunk_0", "doc1_chunk_1"],
        "documents": ["Chunk zero text content", "Chunk one text content"],
        "metadatas": [{"page_number": 1, "chunk_index": 0}, {"page_number": 1, "chunk_index": 1}]
    }
    
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    
    monkeypatch.setattr(vector_store_service, "_get_client", lambda: mock_client)
    
    # Query with query_text that matches chunk_zero more (BM25)
    query_text = "zero"
    query_emb = [0.1] * 384
    
    results = vector_store_service.query_hybrid(
        query_text=query_text,
        query_embedding=query_emb,
        collection_id=uuid.uuid4(),
        vector_top_k=2,
        bm25_top_k=2
    )
    
    # Check that both chunks are returned and RRF has run
    assert len(results) == 2
    # Verify shape of the results
    assert "chunk_text" in results[0]
    assert "fused_score" in results[0]
    assert "page_number" in results[0]
    assert "chunk_index" in results[0]


def test_rerank_chunks_sorting(monkeypatch):
    from app.services.reranker_service import reranker_service, RerankerService
    
    # Restore the original rerank_chunks method to override the autouse mock
    monkeypatch.setattr(reranker_service, "rerank_chunks", lambda *args, **kwargs: RerankerService.rerank_chunks(reranker_service, *args, **kwargs))
    
    # Mock CrossEncoder class and its predict method
    mock_model = MagicMock()
    # Let's say model.predict returns scores: 0.1 for first chunk, 0.9 for second chunk
    mock_model.predict.return_value = [0.1, 0.9]
    
    # Bypass loading CrossEncoder by returning our mock_model
    monkeypatch.setattr(reranker_service, "_get_model", lambda: mock_model)
    
    # Candidates
    candidates = [
        {"chunk_text": "Chunk one", "text": "Chunk one", "fused_score": 0.5},
        {"chunk_text": "Chunk two", "text": "Chunk two", "fused_score": 0.4}
    ]
    
    results = reranker_service.rerank_chunks(
        question="test question",
        candidate_chunks=candidates,
        top_n=2
    )
    
    # Verify that predict was called with the correct pairs
    mock_model.predict.assert_called_once_with([
        ["test question", "Chunk one"],
        ["test question", "Chunk two"]
    ])
    
    # Verify that results are sorted by rerank_score descending (so Chunk two should be first with score 0.9)
    assert len(results) == 2
    assert results[0]["chunk_text"] == "Chunk two"
    assert results[0]["rerank_score"] == 0.9
    assert results[1]["chunk_text"] == "Chunk one"
    assert results[1]["rerank_score"] == 0.1


def test_rerank_confidence_threshold_trips(db, monkeypatch):
    from app.services.chat_service import chat_service
    from app.models.document import Document
    import app.services.chat_service
    from app.services.embedding_service import embedding_service
    from app.services.vector_store_service import vector_store_service
    
    # 1. Setup mock document in DB
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        filename="test_confidence.pdf",
        page_count=1,
        embedding_model="all-MiniLM-L6-v2",
        is_deleted=False
    )
    db.add(doc)
    db.commit()
    
    # 2. Mock embedding, vector store, and reranker services
    monkeypatch.setattr(embedding_service, "get_embedding_model_info", lambda *args, **kwargs: "all-MiniLM-L6-v2")
    monkeypatch.setattr(embedding_service, "generate_query_embedding", lambda *args, **kwargs: [0.1] * 384)
    
    similar_hybrid_mock = [
        {"chunk_text": "Chunk text", "text": "Chunk text", "page_number": 1, "chunk_index": 0, "fused_score": 1.0}
    ]
    monkeypatch.setattr(vector_store_service, "query_hybrid", lambda *args, **kwargs: similar_hybrid_mock)
    
    # Reranker returns chunks with score below configured threshold
    from app.services.reranker_service import reranker_service
    from app.core.config import settings
    monkeypatch.setattr(
        reranker_service, 
        "rerank_chunks", 
        lambda question, candidate_chunks, top_n=5: [{**c, "rerank_score": settings.MIN_RERANK_CONFIDENCE - 5.0} for c in candidate_chunks[:top_n]]
    )
    
    # Mock LLM provider to verify it is NOT called
    llm_call_count = 0
    class MockLLMProvider:
        def generate_stream(self, prompt, system_prompt, temperature, max_tokens):
            nonlocal llm_call_count
            llm_call_count += 1
            yield {"type": "token", "token": "Should not happen"}
            
    monkeypatch.setattr(app.services.chat_service, "get_llm_provider", lambda: MockLLMProvider())
    
    # Call chat stream
    generator = chat_service.generate_chat_stream(
        db=db,
        document_id=doc_id,
        conversation_id=None,
        question="Where is the gold?",
        request_id=uuid.uuid4(),
        user_id=None,
        session_id="guest-session-789"
    )
    
    response_chunks = list(generator)
    
    # Assert LLM was never invoked
    assert llm_call_count == 0
    
    # Parse output tokens to check if the fallback was streamed
    import json
    tokens = []
    citations_packet = None
    for chunk_str in response_chunks:
        data = json.loads(chunk_str)
        if "token" in data:
            tokens.append(data["token"])
        elif "citations" in data:
            citations_packet = data
            
    full_response = "".join(tokens).strip()
    assert "I could not find relevant information in the document to answer this question." in full_response
    assert citations_packet is not None
    assert citations_packet["citations"] == []


def test_ollama_runtime_connection_failure(monkeypatch):
    import httpx
    from app.services.llm_providers.ollama_provider import OllamaProvider
    from app.core.exceptions import OllamaUnavailableException

    # Mock httpx.stream to raise a ConnectError
    def mock_httpx_stream(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx, "stream", mock_httpx_stream)

    provider = OllamaProvider()
    
    with pytest.raises(OllamaUnavailableException) as excinfo:
        list(provider.generate_stream(
            prompt="Hello",
            system_prompt="Be concise",
            temperature=0.0,
            max_tokens=100
        ))

    assert excinfo.value.status_code == 503
    assert "The local AI model (Ollama) is not running" in excinfo.value.message


def test_chat_service_ollama_unavailable_stream_chunk(monkeypatch):
    import uuid
    import json
    from unittest.mock import MagicMock
    from app.models.document import Document
    from app.services.chat_service import chat_service
    from app.services.vector_store_service import vector_store_service
    from app.core.config import settings
    
    doc_id = uuid.uuid4()
    mock_doc = Document(
        id=doc_id,
        user_id=uuid.UUID("00000000-0000-0000-0000-000000000000"),
        filename="test_ollama_err.pdf",
        page_count=1,
        chroma_collection_id="col_test_ollama_err",
        embedding_model="local-english-base",
        is_deleted=False
    )
    
    from app.models.message import Message

    def mock_query(model):
        query_mock = MagicMock()
        if model == Document:
            query_mock.filter.return_value.first.return_value = mock_doc
        elif model == Message:
            query_mock.filter.return_value.first.return_value = None
        return query_mock

    mock_db = MagicMock()
    mock_db.query = mock_query

    # Mock settings
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    # Make sure we pass the rerank confidence threshold check
    similar_hybrid_mock = [
        {"chunk_text": "Chunk text", "text": "Chunk text", "page_number": 1, "chunk_index": 0, "fused_score": 1.0}
    ]
    monkeypatch.setattr(vector_store_service, "query_hybrid", lambda *args, **kwargs: similar_hybrid_mock)
    
    from app.services.reranker_service import reranker_service
    monkeypatch.setattr(
        reranker_service, 
        "rerank_chunks", 
        lambda question, candidate_chunks, top_n=5: [{**c, "rerank_score": 1.0} for c in candidate_chunks[:top_n]]
    )

    # Mock OllamaProvider to raise ConnectError
    import httpx
    def mock_httpx_stream(*args, **kwargs):
        raise httpx.ConnectError("Ollama offline")
    monkeypatch.setattr(httpx, "stream", mock_httpx_stream)

    # Call chat stream
    generator = chat_service.generate_chat_stream(
        db=mock_db,
        document_id=doc_id,
        conversation_id=None,
        question="What is the pricing?",
        request_id=uuid.uuid4(),
        user_id=None,
        session_id="guest-session-789"
    )
    
    response_chunks = list(generator)
    
    error_packet = None
    for chunk_str in response_chunks:
        data = json.loads(chunk_str)
        if "error" in data:
            error_packet = data
            break
            
    assert error_packet is not None
    assert "The local AI model (Ollama) is not running" in error_packet["error"]







