# Docling API

An API for converting and processing various document formats using the Docling library.

## Features

- Convert various document formats (PDF, DOCX, images, etc.) to structured formats
- Extract tables, figures, and text content
- Process documents from both file uploads and URLs
- Export to multiple formats including markdown, JSON, and HTML

## Supported Input Formats

- PDF
- DOCX, XLSX, PPTX
- Markdown
- AsciiDoc
- HTML, XHTML
- CSV
- Images (PNG, JPEG, TIFF, BMP)

## API Endpoints

### `/process/`

Basic document processing with minimal options.

```bash
POST /process/
```

### `/parse/`

Advanced document processing with file upload or form-based URL input.

```bash
POST /parse/
```

### `/parse/url`

JSON-based URL document processing.

```bash
POST /parse/url
```

## Running with Docker

```bash
# Build and start containers
docker-compose up -d

# Stop containers
docker-compose down
```

## Environment Variables

- `MAX_WORKERS`: Control the number of worker processes (default: 4)
- `DOCLING_ARTIFACTS_PATH`: Custom path for Docling models

## Examples

### Processing a URL

```bash
curl -X POST "http://localhost:8081/parse/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "export_figures": true,
    "export_tables": true,
    "export_markdown": true,
    "export_json": true,
    "image_resolution": 2.0
  }'
```

### Uploading a File

```bash
curl -X POST "http://localhost:8081/parse/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf" \
  -F "export_figures=true" \
  -F "export_tables=true" \
  -F "export_markdown=true" \
  -F "export_json=true" \
  -F "image_resolution=2.0"
```

## License

MIT