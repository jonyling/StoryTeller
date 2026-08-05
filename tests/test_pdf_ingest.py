import fitz
import pytest
from PIL import Image

from pipeline.errors import ValidationError
from pipeline.pdf_ingest import MAX_PAGES, downscale, extract_page_images


def _make_pdf_bytes(num_pages: int) -> bytes:
    doc = fitz.open()
    for _ in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), "test page")
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_page_images_returns_one_image_per_page():
    images = extract_page_images(_make_pdf_bytes(4))
    assert len(images) == 4
    assert all(isinstance(image, Image.Image) for image in images)


def test_extract_page_images_rejects_too_many_pages():
    with pytest.raises(ValidationError, match="pages"):
        extract_page_images(_make_pdf_bytes(MAX_PAGES + 1))


def test_extract_page_images_rejects_empty_pdf():
    # Minimal valid PDF with 0 pages (fitz won't save zero-page PDFs, so we create raw bytes)
    empty_pdf_bytes = (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"xref\n0 3\n0000000000 65535 f\n0000000009 00000 n\n0000000074 00000 n\n"
        b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n131\n%%EOF"
    )
    with pytest.raises(ValidationError, match="no pages"):
        extract_page_images(empty_pdf_bytes)


def test_downscale_shrinks_large_image_to_max_dimension():
    large_image = Image.new("RGB", (2000, 1000), color="white")
    result = downscale(large_image, max_dimension=500)
    assert max(result.size) == 500


def test_downscale_leaves_small_image_unchanged():
    small_image = Image.new("RGB", (200, 100), color="white")
    result = downscale(small_image, max_dimension=500)
    assert result.size == (200, 100)
