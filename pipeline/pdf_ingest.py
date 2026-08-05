import fitz
from PIL import Image

from pipeline.errors import ValidationError

MIN_PAGES = 1
MAX_PAGES = 10
RASTER_DPI = 150
MAX_DIMENSION = 1024


def extract_page_images(pdf_bytes: bytes) -> list:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = doc.page_count
        if page_count < MIN_PAGES:
            raise ValidationError("The PDF has no pages.")
        if page_count > MAX_PAGES:
            raise ValidationError(
                f"The PDF has {page_count} pages; please upload {MAX_PAGES} or fewer."
            )
        zoom = RASTER_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        images = []
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(downscale(image))
        return images
    finally:
        doc.close()


def downscale(image, max_dimension: int = MAX_DIMENSION):
    width, height = image.size
    largest = max(width, height)
    if largest <= max_dimension:
        return image
    scale = max_dimension / largest
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)
