#!/usr/bin/env python3
"""
MinerU Document Processing Script
Functions:
1. Copy images parsed by MinerU to the frontend public directory
2. Modify image path references in the MD file
3. Add the processed MD content to the vector store
"""

import os
import shutil
from pathlib import Path
import re
from typing import List
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

MINERU_DIR = Path(__file__).parent / "File_fs"
FRONTEND_PUBLIC_DIR = "agent-chat-ui/public"
VECTORSTORE_PATH = "fs_db"
MD_FILE = "full.md"
IMAGES_DIR = "images"

IMAGE_PREFIX = "/fs-images"

def setup_directories():
    frontend_images_dir = Path(FRONTEND_PUBLIC_DIR) / "fs-images"
    frontend_images_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created frontend image directory: {frontend_images_dir}")
    return frontend_images_dir

def copy_images_to_frontend(source_dir: Path, target_dir: Path) -> List[str]:
    """Copy images to frontend directory"""
    copied_images = []
    source_images_dir = source_dir / IMAGES_DIR
    
    if not source_images_dir.exists():
        print(f"⚠️  Image directory does not exist: {source_images_dir}")
        return copied_images
    
    for image_file in source_images_dir.glob("*.jpg"):
        target_file = target_dir / image_file.name
        shutil.copy2(image_file, target_file)
        copied_images.append(image_file.name)
        print(f"📷 Copied image: {image_file.name}")
    
    print(f"✅ {len(copied_images)} images copied")
    return copied_images

def update_image_paths_in_md(md_content: str) -> str:
    """Update image paths in the MD file"""
    pattern = r'!\[\]\(images/([^)]+)\)'
    
    def replace_image_path(match):
        filename = match.group(1)
        new_path = f"{IMAGE_PREFIX}/{filename}"
        return f"![]({new_path})"
    
    updated_content = re.sub(pattern, replace_image_path, md_content)
    
    original_count = len(re.findall(pattern, md_content))
    print(f"🔄 Updated {original_count} image path references")
    
    return updated_content

def split_markdown_content(content: str) -> List[Document]:
    """Split markdown content into document chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n# ",      # Level 1 heading
            "\n## ",     # Level 2 heading
            "\n### ",    # Level 3 heading
            "\n\n",      # Paragraph
            "\n",        # Line
            " ",         # Word
            ""           # Character
        ]
    )
    
    # Split text
    texts = text_splitter.split_text(content)
    
    # Create Document object
    documents = []
    for i, text in enumerate(texts):
        doc = Document(
            page_content=text,
            metadata={
                "source": "SEC",
                "chunk_id": i,
                "document_type": "FS",
                "processed_by": "MinerU"
            }
        )
        documents.append(doc)
    
    print(f"📄 Split into {len(documents)} chunks")
    return documents

def add_to_vectorstore(documents: List[Document]):
    """Add documents to vector store"""
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
            base_url="https://ai.devtool.tech/proxy/v1",
            model="text-embedding-3-small",
        )
        
        # Check if existing vector store exists
        vectorstore_path = Path(VECTORSTORE_PATH)
        
        if vectorstore_path.exists():
            print("📚 Loading existing vector store...")
            # Load existing vector store
            vectorstore = FAISS.load_local(
                folder_path=VECTORSTORE_PATH,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            
            # Add new documents
            vectorstore.add_documents(documents)
            print("✅ New documents added to existing vector store")
            
        else:
            print("🆕 Creating new vector store...")
            vectorstore = FAISS.from_documents(documents, embeddings)
            print("✅ New vector store created successfully")
        
        # Save vector store
        vectorstore.save_local(VECTORSTORE_PATH)
        print(f"💾 Vector store saved to: {VECTORSTORE_PATH}")
        
        # Display vector store information
        print(f"📊 Total number of documents in vector store: {vectorstore.index.ntotal}")
        
    except Exception as e:
        print(f"❌ Vector store operation failed: {str(e)}")
        raise

def validate_environment():
    """Validate environment configuration"""
    required_keys = ["OPENAI_API_KEY"]  # Please set your API key in .env file
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing required environment variables: {', '.join(missing_keys)}")
        print("Please configure these variables in the .env file")
        return False
    
    print("✅ Environment variables validated successfully")
    return True

def main():
    """Main function"""
    print("🚀 Starting to process MinerU documents...")
    
    # Validate environment
    if not validate_environment():
        return
    
    # Check source file
    source_dir = Path(MINERU_DIR)
    md_file_path = source_dir / MD_FILE 

    if not md_file_path.exists():
        print(f"❌ MD file not found: {md_file_path}")
        return
    
    try:
        # 1. Set directory
        frontend_images_dir = setup_directories()
        
        # 2. Copy images to frontend
        copied_images = copy_images_to_frontend(source_dir, frontend_images_dir)
        
        # 3. Read and process MD file
        print(f"📖 Reading MD file: {md_file_path}")
        with open(md_file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # 4. Update image paths
        updated_content = update_image_paths_in_md(md_content)
        
        # 5. Save updated MD file (optional)
        updated_md_path = source_dir / "full_updated.md"
        with open(updated_md_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"💾 Saved updated MD file: {updated_md_path}")
        
        # 6. Split content
        documents = split_markdown_content(updated_content)
        
        # 7. Add to vector store
        add_to_vectorstore(documents)
        
        print(f"📝 Processing completed!")
        print(f"   - Images copied: {len(copied_images)}")
        print(f"   - Document chunks: {len(documents)}")
        print(f"   - Vector store path: {VECTORSTORE_PATH}")
        print(f"   - Frontend image path: {frontend_images_dir}")
        
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()