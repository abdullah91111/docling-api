from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Depends
from io import BytesIO
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions, VlmPipelineOptions, EasyOcrOptions, smoldocling_vlm_conversion_options
from docling.datamodel.base_models import InputFormat, FigureElement
from docling.document_converter import PdfFormatOption, WordFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from pathlib import Path
import pandas as pd
import json
import base64
import os
import urllib.request
import shutil
from typing import Dict, List, Optional, Union, Any
from docling.pipeline.vlm_pipeline import VlmPipeline

app = FastAPI()

class DoclingRequestOptions(BaseModel):
    export_figures: bool = True
    export_tables: bool = True
    export_markdown: bool = True
    export_json: bool = True
    image_resolution: float = 2.0

class UrlRequest(DoclingRequestOptions):
    url: str

# Helper function to process document and return results (WITHOUT VLM)
# This is the original implementation for standard /parse endpoints
async def process_document_common(buf, filename, options):
    # Configure simple document converter without OCR
    doc_converter = DocumentConverter()
    
    # Create a source with a Path-like object
    source = DocumentStream(name=filename, stream=buf)
    
    # Convert the document
    result = doc_converter.convert(source)
    
    # Prepare response data
    response_data = {
        "document_name": filename,
        "document_type": str(result.input.format),
    }
    
    # Export to markdown if requested
    if options.export_markdown:
        response_data["markdown"] = result.document.export_to_markdown()
        
        # Export markdown with embedded images
        if options.export_figures:
            try:
                # Use string output
                md_with_images = result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
                response_data["markdown_with_embedded_images"] = md_with_images
            except Exception as md_err:
                response_data["markdown_with_embedded_images_error"] = str(md_err)
    
    # Export to JSON if requested
    if options.export_json:
        response_data["json"] = result.document.export_to_dict()
    
    # Export figures if requested
    if options.export_figures:
        page_images = {}
        for page_no, page in result.document.pages.items():
            if hasattr(page, 'image') and page.image and hasattr(page.image, 'pil_image'):
                try:
                    img_io = BytesIO()
                    page.image.pil_image.save(img_io, format="PNG")
                    img_io.seek(0)
                    page_images[str(page_no)] = base64.b64encode(img_io.getvalue()).decode('utf-8')
                except Exception as img_err:
                    page_images[str(page_no)] = f"Error: {str(img_err)}"
        
        if page_images:
            response_data["page_images"] = page_images
        
        # Export picture items
        pictures = []
        picture_counter = 0
        for element, _level in result.document.iterate_items():
            if isinstance(element, PictureItem):
                try:
                    picture_counter += 1
                    img_io = BytesIO()
                    img = element.get_image(result.document)
                    if img:
                        img.save(img_io, format="PNG")
                        img_io.seek(0)
                        pictures.append({
                            "id": picture_counter,
                            "data": base64.b64encode(img_io.getvalue()).decode('utf-8')
                        })
                except Exception as pic_err:
                    pictures.append({
                        "id": picture_counter,
                        "error": str(pic_err)
                    })
        
        if pictures:
            response_data["pictures"] = pictures
    
    # Export tables if requested
    if options.export_tables:
        tables = []
        for table_ix, table in enumerate(result.document.tables):
            try:
                table_df = table.export_to_dataframe()
                tables.append({
                    "id": table_ix + 1,
                    "html": table.export_to_html(),
                    "csv": table_df.to_csv(),
                    "data": table_df.to_dict(orient='records')
                })
            except Exception as table_err:
                tables.append({
                    "id": table_ix + 1,
                    "error": str(table_err)
                })
        
        if tables:
            response_data["tables"] = tables
    
    return response_data

# Helper function to process document with VLM (for /parse/vlm endpoint only)
async def process_document_vlm(buf, filename, options):
    # Configure VLM pipeline (SmolDocling)
    vlm_pipeline_options = VlmPipelineOptions()
    vlm_pipeline_options.force_backend_text = False
    
    # Customize the prompt to improve text extraction - important!
    if hasattr(vlm_pipeline_options, "prompt_template"):
        # If available, set a better prompt template
        vlm_pipeline_options.prompt_template = "Convert this page to docling."
    
    vlm_pipeline_options.vlm_options = smoldocling_vlm_conversion_options
    
    # Configure standard pipeline with OCR
    ocr_options = EasyOcrOptions(
        languages=["en"],
        gpu=True,  # Enable GPU acceleration
        min_size=10,
        text_threshold=0.7
    )
    
    std_pipeline_options = PdfPipelineOptions(
        images_scale=options.image_resolution,
        generate_page_images=options.export_figures,
        generate_picture_images=options.export_figures,
        do_table_structure=options.export_tables,
        ocr_options=ocr_options
    )
    
    # Configure multi-format document converter with both pipelines
    doc_converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.CSV,
            InputFormat.MD,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline, 
                pipeline_options=vlm_pipeline_options
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline
            ),
            InputFormat.IMAGE: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=vlm_pipeline_options
            ),
        },
    )
    
    # Create a source with a Path-like object
    source = DocumentStream(name=filename, stream=buf)
    
    # Convert the document
    result = doc_converter.convert(source)
    
    # Prepare response data
    response_data = {
        "document_name": filename,
        "document_type": str(result.input.format),
    }
    
    # Add direct access to VLM prediction text for images
    raw_vlm_text = []
    if str(result.input.format) == "InputFormat.IMAGE":
        # Get raw VLM text predictions from each page
        for page_idx, page in enumerate(result.pages):
            if hasattr(page, 'predictions') and hasattr(page.predictions, 'vlm_response'):
                # Extract the raw doctags text
                doctags = page.predictions.vlm_response.text
                raw_vlm_text.append(doctags)
                
                # Try to create DocTagsDocument for better processing if possible
                try:
                    from docling_core.types.doc import DocTagsDocument, DoclingDocument
                    from PIL import Image
                    
                    # Create Docling document from doctags
                    img = None
                    if hasattr(page, 'image') and hasattr(page.image, 'pil_image'):
                        img = page.image.pil_image
                    
                    if img and doctags:
                        # Create a proper Docling document
                        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [img])
                        doc = DoclingDocument(name=f"{filename}_page_{page_idx+1}")
                        doc.load_from_doctags(doctags_doc)
                        
                        # Get markdown from the document
                        md_text = doc.export_to_markdown()
                        if md_text and len(md_text.strip()) > 0:
                            # Replace the doctags with the cleaner markdown
                            raw_vlm_text[-1] = md_text
                except Exception as e:
                    # If conversion fails, keep the raw doctags
                    print(f"Error converting doctags to markdown: {str(e)}")
        
        # Add combined raw text to the response
        if raw_vlm_text:
            combined_text = "\n\n".join(raw_vlm_text)
            response_data["extracted_text"] = combined_text
            
            # Also update the markdown output to include the extracted text
            if options.export_markdown:
                response_data["markdown"] = combined_text
                response_data["markdown_with_embedded_images"] = combined_text
    else:
        # Normal markdown extraction for non-image files
        if options.export_markdown:
            response_data["markdown"] = result.document.export_to_markdown()
            
            # Export markdown with embedded images
            if options.export_figures:
                try:
                    # Use string output
                    md_with_images = result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
                    response_data["markdown_with_embedded_images"] = md_with_images
                except Exception as md_err:
                    response_data["markdown_with_embedded_images_error"] = str(md_err)
    
    # Export to JSON if requested
    if options.export_json:
        response_data["json"] = result.document.export_to_dict()
        
        # For image files, add the raw text to the JSON export
        if str(result.input.format) == "InputFormat.IMAGE" and raw_vlm_text:
            json_data = response_data["json"]
            if isinstance(json_data, dict):
                json_data["extracted_text"] = "\n\n".join(raw_vlm_text)
                response_data["json"] = json_data
    
    # Export figures if requested
    if options.export_figures:
        page_images = {}
        for page_no, page in result.document.pages.items():
            if hasattr(page, 'image') and page.image and hasattr(page.image, 'pil_image'):
                try:
                    img_io = BytesIO()
                    page.image.pil_image.save(img_io, format="PNG")
                    img_io.seek(0)
                    page_images[str(page_no)] = base64.b64encode(img_io.getvalue()).decode('utf-8')
                except Exception as img_err:
                    page_images[str(page_no)] = f"Error: {str(img_err)}"
        
        if page_images:
            response_data["page_images"] = page_images
        
        # Export picture items
        pictures = []
        picture_counter = 0
        for element, _level in result.document.iterate_items():
            if isinstance(element, PictureItem):
                try:
                    picture_counter += 1
                    img_io = BytesIO()
                    img = element.get_image(result.document)
                    if img:
                        img.save(img_io, format="PNG")
                        img_io.seek(0)
                        pictures.append({
                            "id": picture_counter,
                            "data": base64.b64encode(img_io.getvalue()).decode('utf-8')
                        })
                except Exception as pic_err:
                    pictures.append({
                        "id": picture_counter,
                        "error": str(pic_err)
                    })
        
        if pictures:
            response_data["pictures"] = pictures
    
    # Export tables if requested
    if options.export_tables:
        tables = []
        for table_ix, table in enumerate(result.document.tables):
            try:
                table_df = table.export_to_dataframe()
                tables.append({
                    "id": table_ix + 1,
                    "html": table.export_to_html(),
                    "csv": table_df.to_csv(),
                    "data": table_df.to_dict(orient='records')
                })
            except Exception as table_err:
                tables.append({
                    "id": table_ix + 1,
                    "error": str(table_err)
                })
        
        if tables:
            response_data["tables"] = tables
    
    return response_data

@app.get("/")
def read_root():
    return {"message": "Docling API is up and running!"}

@app.post("/process/")
async def process_data(file: UploadFile = File(...)):
    try:
        # Read the uploaded file into memory as binary stream
        pdf_file = await file.read()
        buf = BytesIO(pdf_file)
        
        # Convert the uploaded PDF using Docling
        source = DocumentStream(name=file.filename, stream=buf)
        
        # Configure the document converter options (custom options can be added)
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        # Convert the document
        result = doc_converter.convert(source)
        
        # Export the converted document to markdown (you can change the format as needed)
        markdown_output = result.document.export_to_markdown()
        
        return {"converted_document": markdown_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

# Helper function to parse options from form data
async def get_form_options(
    export_figures: Optional[str] = Form(None),
    export_tables: Optional[str] = Form(None),
    export_markdown: Optional[str] = Form(None),
    export_json: Optional[str] = Form(None),
    image_resolution: Optional[str] = Form(None),
) -> DoclingRequestOptions:
    options = DoclingRequestOptions()
    
    # Parse form values to proper types
    if export_figures is not None:
        options.export_figures = export_figures.lower() == 'true'
    if export_tables is not None:
        options.export_tables = export_tables.lower() == 'true'
    if export_markdown is not None:
        options.export_markdown = export_markdown.lower() == 'true'
    if export_json is not None:
        options.export_json = export_json.lower() == 'true'
    if image_resolution is not None:
        try:
            options.image_resolution = float(image_resolution)
        except ValueError:
            pass  # Keep default if not a valid float
    
    return options

# Form-based endpoint for file uploads - standard OCR without VLM
@app.post("/parse/")
async def parse_document_form(
    file: Optional[UploadFile] = File(None),
    url_form: Optional[str] = Form(None),
    form_options: DoclingRequestOptions = Depends(get_form_options)
):
    try:
        # Check if either file or url is provided in form
        if file is None and not url_form:
            raise HTTPException(
                status_code=400, 
                detail="Either file upload or URL must be provided in form data"
            )
            
        # Handle data source (file upload or URL)
        if file:
            # Process uploaded file
            content = await file.read()
            buf = BytesIO(content)
            filename = file.filename
        else:
            # Process URL from form
            try:
                # Fetch content from URL
                with urllib.request.urlopen(url_form) as response:
                    content = response.read()
                    buf = BytesIO(content)
                    # Extract filename from URL
                    filename = os.path.basename(url_form)
                    if not filename:
                        filename = "document_from_url"
            except Exception as url_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to fetch document from URL: {str(url_err)}"
                )
                
        # Process the document using the common function (without VLM)
        return await process_document_common(buf, filename, form_options)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing document: {str(e)}")

# JSON-based endpoint for URL processing - standard OCR without VLM
@app.post("/parse/url")
async def parse_document_url(request: UrlRequest):
    try:
        url_to_use = request.url
        
        # Check that URL is provided
        if not url_to_use:
            raise HTTPException(
                status_code=400, 
                detail="URL must be provided"
            )
            
        # Process URL from JSON body
        try:
            # Fetch content from URL
            with urllib.request.urlopen(url_to_use) as response:
                content = response.read()
                buf = BytesIO(content)
                # Extract filename from URL
                filename = os.path.basename(url_to_use)
                if not filename:
                    filename = "document_from_url"
        except Exception as url_err:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to fetch document from URL: {str(url_err)}"
            )
                
        # Process the document using the common function (without VLM)
        return await process_document_common(buf, filename, request)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing document: {str(e)}")

# VLM endpoint with advanced OCR and text extraction
@app.post("/parse/vlm")
async def parse_document_vlm(
    file: Optional[UploadFile] = File(None),
    url_form: Optional[str] = Form(None),
    form_options: DoclingRequestOptions = Depends(get_form_options)
):
    try:
        # Same logic for handling file/URL as in parse_document_form
        if file is None and not url_form:
            raise HTTPException(
                status_code=400, 
                detail="Either file upload or URL must be provided in form data"
            )
            
        if file:
            content = await file.read()
            buf = BytesIO(content)
            filename = file.filename
        else:
            try:
                with urllib.request.urlopen(url_form) as response:
                    content = response.read()
                    buf = BytesIO(content)
                    filename = os.path.basename(url_form)
                    if not filename:
                        filename = "document_from_url"
            except Exception as url_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to fetch document from URL: {str(url_err)}"
                )
        
        # Process the document using the VLM function
        return await process_document_vlm(buf, filename, form_options)
        
    except Exception as e:
        # Try to fix the SmolDocling config if needed
        if "config file" in str(e) and "is not a valid JSON file" in str(e):
            if verify_and_fix_model_config():
                # Try again after fixing the config
                return await parse_document_vlm(file, url_form, form_options)
                
        raise HTTPException(status_code=500, detail=f"Error parsing document with VLM: {str(e)}")

# Add function to check and fix SmolDocling model config
def verify_and_fix_model_config():
    """
    Verifies the SmolDocling model configuration file and attempts to fix it if corrupted.
    Returns True if successful, False otherwise.
    """
    try:
        # Define expected path patterns for SmolDocling model
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        model_dirs = Path(cache_dir).glob("hub/models--ds4sd--SmolDocling*")
        
        for model_dir in model_dirs:
            snapshot_dirs = list(model_dir.glob("snapshots/*"))
            
            for snapshot_dir in snapshot_dirs:
                config_path = snapshot_dir / "processor_config.json"
                
                if config_path.exists():
                    # Try to read the JSON file
                    try:
                        with open(config_path, 'r') as f:
                            json.loads(f.read())
                        print(f"Config file at {config_path} is valid.")
                    except json.JSONDecodeError:
                        print(f"Found invalid config at {config_path}, attempting to fix...")
                        
                        # Create backup
                        backup_path = config_path.with_suffix('.json.bak')
                        shutil.copy2(config_path, backup_path)
                        
                        # Download a fresh copy or create a valid config
                        try:
                            # Try to download from Hugging Face
                            url = "https://huggingface.co/ds4sd/SmolDocling-256M-preview/raw/main/processor_config.json"
                            urllib.request.urlretrieve(url, config_path)
                            print(f"Successfully downloaded new config file to {config_path}")
                            return True
                        except Exception as download_err:
                            print(f"Failed to download config: {download_err}")
                            
                            # Create minimal valid config as fallback
                            default_config = {
                                "feature_extractor_type": "DonutFeatureExtractor",
                                "processor_class": "DonutProcessor",
                                "tokenizer_class": "PreTrainedTokenizerFast"
                            }
                            
                            with open(config_path, 'w') as f:
                                json.dump(default_config, f, indent=2)
                            print(f"Created default config file at {config_path}")
                            return True
        
        return False
    except Exception as e:
        print(f"Error verifying/fixing model config: {e}")
        return False

# Call verify_and_fix_model_config on startup
@app.on_event("startup")
async def startup_event():
    verify_and_fix_model_config()