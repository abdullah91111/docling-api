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
    # Configure multi-format document converter (standard OCR pipeline)
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
                pipeline_cls=StandardPdfPipeline, 
                backend=PyPdfiumDocumentBackend,
                pipeline_options=PdfPipelineOptions(
                    images_scale=options.image_resolution,
                    generate_page_images=options.export_figures,
                    generate_picture_images=options.export_figures,
                    do_table_structure=options.export_tables
                )
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline
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
                        doc = DocTagsDocument.from_doctags(doctags)
                        if doc:
                            response_data[f"page_{page_idx}_structured"] = doc.export_to_dict()
                except Exception as doctags_err:
                    # Failed to create structured document - include raw text only
                    pass
    
    # Include raw VLM text in response if available
    if raw_vlm_text:
        response_data["vlm_raw_text"] = raw_vlm_text
    
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
        for page_idx, page in enumerate(result.pages):
            if hasattr(page, 'image') and page.image and hasattr(page.image, 'pil_image'):
                try:
                    img_io = BytesIO()
                    page.image.pil_image.save(img_io, format="PNG")
                    img_io.seek(0)
                    page_images[str(page_idx)] = base64.b64encode(img_io.getvalue()).decode('utf-8')
                except Exception as img_err:
                    page_images[str(page_idx)] = f"Error: {str(img_err)}"
        
        if page_images:
            response_data["page_images"] = page_images
        
        # Export picture items (only for standard document)
        if hasattr(result, 'document'):
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
    if options.export_tables and hasattr(result, 'document'):
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
    return {"status": "ok", "message": "Docling API is running"}

@app.post("/process/")
async def process_data(file: UploadFile = File(...)):
    try:
        # Get file content as bytes
        file_bytes = await file.read()
        file_name = file.filename
        
        # Process with default options
        options = DoclingRequestOptions()
        result = await process_document_common(BytesIO(file_bytes), file_name, options)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper to get form options
async def get_form_options(
    export_figures: Optional[str] = Form(None),
    export_tables: Optional[str] = Form(None),
    export_markdown: Optional[str] = Form(None),
    export_json: Optional[str] = Form(None),
    image_resolution: Optional[str] = Form(None),
) -> DoclingRequestOptions:
    options = DoclingRequestOptions()
    
    # Convert string form values to appropriate types
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
            pass  # Use default if conversion fails
    
    return options

@app.post("/parse/")
async def parse_document_form(
    file: Optional[UploadFile] = File(None),
    url_form: Optional[str] = Form(None),
    form_options: DoclingRequestOptions = Depends(get_form_options)
):
    try:
        # Process file upload
        if file:
            file_bytes = await file.read()
            file_name = file.filename
            result = await process_document_common(BytesIO(file_bytes), file_name, form_options)
            return result
        
        # Process URL
        elif url_form:
            # Download file from URL
            with urllib.request.urlopen(url_form) as response:
                file_content = response.read()
            
            # Get filename from URL
            file_name = os.path.basename(url_form)
            if not file_name:
                file_name = "downloaded_file"
            
            result = await process_document_common(BytesIO(file_content), file_name, form_options)
            return result
        
        else:
            raise HTTPException(status_code=400, detail="No file or URL provided")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parse/url")
async def parse_document_url(request: UrlRequest):
    try:
        # Download file from URL
        with urllib.request.urlopen(request.url) as response:
            file_content = response.read()
        
        # Get filename from URL
        file_name = os.path.basename(request.url)
        if not file_name:
            file_name = "downloaded_file"
        
        # Convert options from Pydantic model
        result = await process_document_common(BytesIO(file_content), file_name, request)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parse/vlm")
async def parse_document_vlm(
    file: Optional[UploadFile] = File(None),
    url_form: Optional[str] = Form(None),
    form_options: DoclingRequestOptions = Depends(get_form_options)
):
    try:
        # Process file upload
        if file:
            file_bytes = await file.read()
            file_name = file.filename
            result = await process_document_vlm(BytesIO(file_bytes), file_name, form_options)
            return result
        
        # Process URL
        elif url_form:
            # Download file from URL
            with urllib.request.urlopen(url_form) as response:
                file_content = response.read()
            
            # Get filename from URL
            file_name = os.path.basename(url_form)
            if not file_name:
                file_name = "downloaded_file"
            
            result = await process_document_vlm(BytesIO(file_content), file_name, form_options)
            return result
        
        else:
            raise HTTPException(status_code=400, detail="No file or URL provided")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Verify model configs during startup
def verify_and_fix_model_config():
    """Verify model configuration and optimize if possible"""
    
    try:
        # Check if transformers is available
        import importlib.util
        has_transformers = importlib.util.find_spec("transformers") is not None
        
        if has_transformers:
            # Verify model config
            import torch
            from transformers import AutoConfig
            
            # Check if we have a GPU
            has_gpu = torch.cuda.is_available()
            has_mps = hasattr(torch, 'mps') and torch.backends.mps.is_available()
            
            # Set up model config to be efficient
            if has_gpu:
                print("🔥 GPU detected, will use for inference if needed")
                torch.backends.cudnn.benchmark = True
            elif has_mps:
                print("🍎 Apple Silicon detected, will use MPS if needed")
            else:
                print("⚠️ No GPU detected, performance may be limited for VLM")
            
            # Return detected capabilities
            return {
                "has_transformers": True,
                "has_gpu": has_gpu,
                "has_mps": has_mps,
                "torch_version": torch.__version__
            }
    except Exception as e:
        print(f"⚠️ Error during model verification: {e}")
    
    # Default fallback
    return {
        "has_transformers": False,
        "has_gpu": False,
        "has_mps": False
    }

@app.on_event("startup")
async def startup_event():
    """Prepare models on startup"""
    
    # Verify model configuration
    model_config = verify_and_fix_model_config()
    print(f"🚀 Docling API started with config: {model_config}")
