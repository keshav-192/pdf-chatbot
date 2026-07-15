import logging
import re
from typing import Tuple, List, Dict, Any
from app.core.exceptions import PasswordProtectedException, PageLimitExceededException

logger = logging.getLogger("app.services.pdf_parser")

# Lazy imports for OCR dependencies to handle environments without tesseract binaries
try:
    import fitz  # PyMuPDF
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    HAS_OCR_DEPENDENCIES = True
except ImportError as e:
    logger.warning(f"OCR libraries missing or PyMuPDF error: {e}. OCR fallback will be disabled.")
    HAS_OCR_DEPENDENCIES = False

def clean_extracted_text(text: str) -> str:
    """
    Cleans extracted text by stripping excessive whitespace and duplicate newlines.
    """
    if not text:
        return ""
    # Standardize horizontal spacing
    text = re.sub(r"[ \t]+", " ", text)
    # Standardize vertical spacing (max two consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# OLD: parsed PDF page text using only PyMuPDF (fitz), without column detection or table extraction — replaced below to add pdfplumber-based column/table detection and extraction
# def parse_pdf_pages(file_bytes: bytes) -> Tuple[List[Dict[str, Any]], bool]:
#     """
#     Parses a PDF file from bytes page-by-page.
#     Supports password checks, page count validation, and OCR fallback.
#     """
#     try:
#         import fitz
#     except ImportError:
#         raise RuntimeError("PyMuPDF (fitz) is not installed in the virtual environment.")
# 
#     doc = fitz.open(stream=file_bytes, filetype="pdf")
#     try:
#         # 1. Check if the document is password-protected
#         if doc.is_encrypted:
#             # Try authenticating with empty password
#             if not doc.authenticate(""):
#                 raise PasswordProtectedException("Password protected PDF files are not supported.")
# 
#         # 2. Check page limit constraint (100 pages maximum)
#         page_count = len(doc)
#         if page_count > 100:
#             raise PageLimitExceededException(f"PDF page count ({page_count}) exceeds allowable limit of 100 pages.")
# 
#         pages_data = []
#         ocr_triggered = False
# 
#         for page_idx in range(page_count):
#             page = doc[page_idx]
#             extracted_text = page.get_text()
#             cleaned_text = clean_extracted_text(extracted_text)
# 
#             # Determine if OCR fallback should trigger:
#             # - Extracted text character length is < 100
#             # - Page has drawings, images or measurable dimensions (not completely empty)
#             has_content_visuals = len(page.get_images()) > 0 or len(page.get_drawings()) > 0
#             if len(cleaned_text) < 100 and has_content_visuals:
#                 if HAS_OCR_DEPENDENCIES:
#                     try:
#                         logger.info(f"Triggering OCR fallback for page {page_idx + 1} (chars: {len(cleaned_text)})")
#                         # Convert only this specific page to image
#                         images = convert_from_bytes(file_bytes, first_page=page_idx + 1, last_page=page_idx + 1)
#                         if images:
#                             ocr_text = pytesseract.image_to_string(images[0])
#                             cleaned_ocr = clean_extracted_text(ocr_text)
#                             if len(cleaned_ocr) > len(cleaned_text):
#                                 cleaned_text = cleaned_ocr
#                                 ocr_triggered = True
#                                 logger.info(f"OCR successfully extracted text for page {page_idx + 1}.")
#                     except Exception as ocr_err:
#                         # Log error and fallback to standard text (graceful degradation)
#                         logger.warning(f"OCR execution failed on page {page_idx + 1}: {ocr_err}. Falling back to default.")
#                 else:
#                     logger.warning(f"OCR libraries not fully loaded. Skipping OCR check for page {page_idx + 1}.")
# 
#             pages_data.append({
#                 "page_number": page_idx + 1,
#                 "text": cleaned_text,
#                 "char_count": len(cleaned_text)
#             })
# 
#         return pages_data, ocr_triggered
#     finally:
#         doc.close()

import io
import pdfplumber

# Silencing pdfminer and pdfplumber loggers to prevent debug logging loops
logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("pdfplumber").setLevel(logging.WARNING)
logging.getLogger("pdfminer").propagate = False
logging.getLogger("pdfplumber").propagate = False

def parse_pdf_pages(file_bytes: bytes) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Parses a PDF file from bytes page-by-page.
    Supports password checks, page count validation, and OCR fallback.
    Uses pdfplumber for multi-column layout detection/splitting and table extraction.
    """
    try:
        import fitz
    except ImportError:
        raise RuntimeError("PyMuPDF (fitz) is not installed in the virtual environment.")

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pdf_plumber_client = None
    try:
        # 1. Check if the document is password-protected
        if doc.is_encrypted:
            # Try authenticating with empty password
            if not doc.authenticate(""):
                raise PasswordProtectedException("Password protected PDF files are not supported.")

        # 2. Check page limit constraint (100 pages maximum)
        page_count = len(doc)
        if page_count > 100:
            raise PageLimitExceededException(f"PDF page count ({page_count}) exceeds allowable limit of 100 pages.")

        pages_data = []
        ocr_triggered = False

        # Open pdfplumber for column/table processing
        try:
            pdf_plumber_client = pdfplumber.open(io.BytesIO(file_bytes))
        except Exception as plumber_err:
            logger.warning(f"Could not initialize pdfplumber client: {plumber_err}. Column/table extraction will be disabled.")
            pdf_plumber_client = None

        for page_idx in range(page_count):
            page = doc[page_idx]
            page_width = page.rect.width
            mid_x = page_width / 2

            # Try column detection on PyMuPDF layout blocks
            blocks = page.get_text("blocks")
            is_multi_column = False
            text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
            if len(text_blocks) >= 4:
                left_cnt = 0
                right_cnt = 0
                spanning_cnt = 0
                for x0, y0, x1, y1, txt, b_no, b_type in text_blocks:
                    if x1 <= mid_x + 15:
                        left_cnt += 1
                    elif x0 >= mid_x - 15:
                        right_cnt += 1
                    else:
                        spanning_cnt += 1
                if left_cnt >= 2 and right_cnt >= 2 and spanning_cnt <= (0.25 * len(text_blocks)):
                    is_multi_column = True

            # Perform text extraction
            extracted_text = ""
            was_multi_column = False
            had_tables = False

            plumber_page = None
            if pdf_plumber_client and page_idx < len(pdf_plumber_client.pages):
                plumber_page = pdf_plumber_client.pages[page_idx]

            # If multi-column detected, use pdfplumber split extraction
            if is_multi_column and plumber_page:
                try:
                    logger.info(f"Multi-column layout detected on page {page_idx + 1}. Using pdfplumber column split extraction.")
                    w = plumber_page.width
                    h = plumber_page.height
                    pmid_x = w / 2
                    
                    left_crop = plumber_page.within_bbox((0, 0, pmid_x, h))
                    left_text = left_crop.extract_text() or ""
                    
                    right_crop = plumber_page.within_bbox((pmid_x, 0, w, h))
                    right_text = right_crop.extract_text() or ""
                    
                    extracted_text = left_text + "\n\n" + right_text
                    was_multi_column = True
                except Exception as col_err:
                    logger.warning(f"pdfplumber column extraction failed on page {page_idx + 1}: {col_err}. Falling back to PyMuPDF.")
                    extracted_text = page.get_text()
            else:
                extracted_text = page.get_text()

            # Clean the extracted text
            cleaned_text = clean_extracted_text(extracted_text)

            # Table extraction using pdfplumber
            table_natural_text = ""
            if plumber_page:
                try:
                    tables = plumber_page.extract_tables()
                    if tables:
                        had_tables = True
                        table_lines = []
                        for tbl_idx, table in enumerate(tables):
                            if not table or not table[0]:
                                continue
                            headers = [str(h or f"Column {i+1}").strip() for i, h in enumerate(table[0])]
                            table_lines.append(f"Table {tbl_idx + 1}:")
                            for r_idx, row in enumerate(table[1:]):
                                row_items = []
                                for c_idx, val in enumerate(row):
                                    col_name = headers[c_idx] if c_idx < len(headers) else f"Column {c_idx+1}"
                                    val_str = str(val or "").strip()
                                    row_items.append(f"{col_name} is {val_str}")
                                table_lines.append(f"Row {r_idx + 1}: " + ", ".join(row_items))
                        table_natural_text = "\n".join(table_lines)
                except Exception as tbl_err:
                    logger.warning(f"pdfplumber table extraction failed on page {page_idx + 1}: {tbl_err}")

            # Append flattened table content to cleaned page text
            if table_natural_text:
                if cleaned_text:
                    cleaned_text = cleaned_text + "\n\n" + table_natural_text
                else:
                    cleaned_text = table_natural_text

            # Determine if OCR fallback should trigger:
            # - Extracted text character length is < 100
            # - Page has drawings, images or measurable dimensions (not completely empty)
            has_content_visuals = len(page.get_images()) > 0 or len(page.get_drawings()) > 0
            # OLD: OCR always ran with default language (eng) only — replaced below to support multilingual OCR based on eng+hin+spa+fra+deu and log warning if output is still short
            # if len(cleaned_text) < 100 and has_content_visuals:
            #     if HAS_OCR_DEPENDENCIES:
            #         try:
            #             logger.info(f"Triggering OCR fallback for page {page_idx + 1} (chars: {len(cleaned_text)})")
            #             # Convert only this specific page to image
            #             images = convert_from_bytes(file_bytes, first_page=page_idx + 1, last_page=page_idx + 1)
            #             if images:
            #                 ocr_text = pytesseract.image_to_string(images[0])
            #                 cleaned_ocr = clean_extracted_text(ocr_text)
            #                 if len(cleaned_ocr) > len(cleaned_text):
            #                     cleaned_text = cleaned_ocr
            #                     ocr_triggered = True
            #                     logger.info(f"OCR successfully extracted text for page {page_idx + 1}.")
            #         except Exception as ocr_err:
            #             # Log error and fallback to standard text (graceful degradation)
            #             logger.warning(f"OCR execution failed on page {page_idx + 1}: {ocr_err}. Falling back to default.")
            #     else:
            #         logger.warning(f"OCR libraries not fully loaded. Skipping OCR check for page {page_idx + 1}.")

            if len(cleaned_text) < 100 and has_content_visuals:
                if HAS_OCR_DEPENDENCIES:
                    try:
                        logger.info(f"Triggering OCR fallback for page {page_idx + 1} (chars: {len(cleaned_text)})")
                        # Convert only this specific page to image
                        images = convert_from_bytes(file_bytes, first_page=page_idx + 1, last_page=page_idx + 1)
                        if images:
                            ocr_text = pytesseract.image_to_string(images[0], lang="eng+hin+spa+fra+deu")
                            cleaned_ocr = clean_extracted_text(ocr_text)
                            
                            # Log warning if the OCR text is still suspiciously short after the multi-language attempt
                            if len(cleaned_ocr) < 100:
                                logger.warning(
                                    f"OCR output on page {page_idx + 1} is suspiciously short ({len(cleaned_ocr)} chars). "
                                    f"OCR may have failed to recognize the document's language properly."
                                )
                            
                            if len(cleaned_ocr) > len(cleaned_text):
                                cleaned_text = cleaned_ocr
                                ocr_triggered = True
                                logger.info(f"OCR successfully extracted text for page {page_idx + 1}.")
                    except Exception as ocr_err:
                        # Log error and fallback to standard text (graceful degradation)
                        logger.warning(f"OCR execution failed on page {page_idx + 1}: {ocr_err}. Falling back to default.")
                else:
                    logger.warning(f"OCR libraries not fully loaded. Skipping OCR check for page {page_idx + 1}.")

            pages_data.append({
                "page_number": page_idx + 1,
                "text": cleaned_text,
                "char_count": len(cleaned_text),
                "was_multi_column": was_multi_column,
                "had_tables": had_tables
            })

        return pages_data, ocr_triggered
    finally:
        doc.close()
        if pdf_plumber_client:
            try:
                pdf_plumber_client.close()
            except Exception:
                pass

