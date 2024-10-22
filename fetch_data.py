from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any
import numpy as np
from git import Repo
import os
import logging
import tiktoken
import shutil
import pandas as pd
import json




class ReadmeChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the ReadmeChunker with customizable chunk parameters.
        
        Args:
            chunk_size (int): The size of each chunk in characters
            chunk_overlap (int): The overlap between chunks in characters
        """

        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=token_length,
            is_separator_regex=False
        )

    def clean_directory(self, directory: str) -> None:
        """
        Safely remove a directory and its contents if it exists.
        
        Args:
            directory (str): Path to the directory to clean
        """
        try:
            if os.path.exists(directory):
                logging.info(f"Removing existing directory: {directory}")
                shutil.rmtree(directory)
        except Exception as e:
            logging.error(f"Error cleaning directory {directory}: {str(e)}")
            raise


    def clone_repo(self, repo_url: str, target_dir: str) -> str:
        """
        Clone a git repository to the specified directory.
        
        Args:
            repo_url (str): URL of the git repository
            target_dir (str): Local directory to clone into
            
        Returns:
            str: Path to the cloned repository
        """
        # Clean existing directory
        self.clean_directory(target_dir)
        
        # Create fresh directory
        os.makedirs(target_dir, exist_ok=True)


        logging.info(f"Cloning repository: {repo_url}")
        Repo.clone_from(repo_url, target_dir)
        return target_dir

    def find_readme_files(self, repo_path: str) -> List[str]:
        """
        Find all README files in the repository.
        
        Args:
            repo_path (str): Path to the repository
            
        Returns:
            List[str]: List of paths to README files
        """
        readme_files = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.lower().startswith('readme.') and file.lower().endswith(('.md', '.txt')):
                    readme_files.append(os.path.join(root, file))
        return readme_files

    def process_readme(self, file_path: str, target_dir: str) -> Document:
        """
        Process a single README file into a Document with metadata.
        
        Args:
            file_path (str): Path to the README file
            target_dir (str): Path of the cloned project
            
        Returns:
            Document: LangChain Document with content and metadata
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract metadata
        metadata = {
            'filename': os.path.basename(file_path),
            'directory': os.path.dirname(file_path).split(f"{target_dir}")[1].replace('/', '-'),
            'file_type': os.path.splitext(file_path)[1],
            'size': token_length(content)
        }

        if metadata['directory'] == '':
            metadata['directory'] = 'main page'
        elif metadata['directory'].startswith('-'):
            metadata['directory'] = metadata['directory'][1:]
        
        return Document(page_content=content, metadata=metadata)

    def chunk_document(self, doc: Document) -> List[Document]:
        """
        Split a document into chunks while preserving metadata.
        
        Args:
            doc (Document): Input document to split
            
        Returns:
            List[Document]: List of chunk documents with preserved metadata
        """
        chunks = self.splitter.split_text(doc.page_content)
        chunk_docs = []
        
        for i, chunk in enumerate(chunks):
            # Create new metadata dict with chunk information
            chunk_metadata = doc.metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_size': token_length(chunk),
                'total_chunks': len(chunks)
            })
            
            chunk_docs.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))
            
        return chunk_docs

    def process_repository(self, repo_url: str, target_dir: str) -> List[Document]:
        """
        Process an entire repository, chunking all README files.
        
        Args:
            repo_url (str): URL of the git repository
            target_dir (str): Local directory to clone into
            
        Returns:
            List[Document]: List of all chunks from all README files
        """
        # Clone the repository
        repo_path = self.clone_repo(repo_url, target_dir)
        
        # Find all README files
        readme_files = self.find_readme_files(repo_path)
        all_chunks = []
        
        # Process each README file
        for file_path in readme_files:
            doc = self.process_readme(file_path, target_dir)
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            
        return all_chunks
    

    def filter_chunks(self, 
                    chunks: List[Document], 
                    target: str,
                        ) -> List[Document]:
        """
        Filter chunks based on specified criteria.
        
        Args:
            chunks (List[Document]): List of document chunks to filter
            target (str): string to search into filename
        Returns:
            List[Document]: Filtered list of chunks
        """

        
        filtered_chunks = []
        removed_count = 0
        
        for chunk in chunks:
            
            
            if target not in chunk.metadata['filename']:        
                filtered_chunks.append(chunk)
            else:
                removed_count += 1
        
        logging.info(f"Filtered {removed_count} chunks out of {len(chunks)} total chunks")
        
        return filtered_chunks



def token_length(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
    return len(enc.encode(text))


def export_docs_to_csv(docs: List[Document], 
                      output_path: str,
                      flatten_metadata: bool = True,
                      max_content_length: int = None) -> str:
    """
    Export LangChain documents to CSV, preserving both content and metadata.
    
    Args:
        docs (List[Document]): List of LangChain documents
        output_path (str): Path where CSV will be saved
        flatten_metadata (bool): If True, creates separate columns for metadata fields
        max_content_length (int): Optional max length for content field
        
    Returns:
        str: Path to the saved CSV file
    """
    # Initialize list to store document data
    rows = []
    
    # Track all unique metadata keys if flattening
    metadata_keys = set()
    if flatten_metadata:
        for doc in docs:
            metadata_keys.update(doc.metadata.keys())
    
    for doc in docs:
        # Start with the content
        row = {}
        
        # Handle content, optionally truncating
        content = doc.page_content
        if max_content_length and len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        row['content'] = content
        
        if flatten_metadata:
            # Add each metadata field as a separate column
            for key in metadata_keys:
                row[f'metadata_{key}'] = doc.metadata.get(key, None)
        else:
            # Store metadata as a JSON string
            row['metadata'] = json.dumps(doc.metadata)
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Optimize string columns
    for col in df.select_dtypes(['object']):
        df[col] = df[col].astype('string')
    
    try:
        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        logging.info(f"Successfully exported {len(docs)} documents to {output_path}")
        
        # Print summary
        print(f"\nExport Summary:")
        print(f"Total documents: {len(docs)}")
        print(f"Total columns: {len(df.columns)}")

        if flatten_metadata:
            print(f"Metadata fields: {', '.join(metadata_keys)}")
        print(f"File saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logging.error(f"Error exporting to CSV: {str(e)}")
        raise

def read_docs_from_csv(csv_path: str, 
                      flatten_metadata: bool = True) -> List[Document]:
    """
    Read LangChain documents from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        flatten_metadata (bool): If True, assumes metadata is in separate columns
        
    Returns:
        List[Document]: List of LangChain documents
    """
    try:
        df = pd.read_csv(csv_path)
        docs = []
        
        for _, row in df.iterrows():
            if flatten_metadata:
                # Collect metadata from columns starting with 'metadata_'
                metadata = {}
                for col in df.columns:
                    if col.startswith('metadata_'):
                        key = col[9:]  # Remove 'metadata_' prefix
                        metadata[key] = row[col]
            else:
                # Parse metadata JSON string
                metadata = json.loads(row['metadata'])
            
            doc = Document(
                page_content=row['content'],
                metadata=metadata
            )
            docs.append(doc)
            
        logging.info(f"Successfully loaded {len(docs)} documents from {csv_path}")
        return docs
        
    except Exception as e:
        logging.error(f"Error reading from CSV: {str(e)}")
        raise



# Example usage
if __name__ == "__main__":

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    chunker = ReadmeChunker(chunk_size=2_000, chunk_overlap=250)

    # Example repository URL and target directory
    repo_url = "https://github.com/microsoft/AI-For-Beginners"
    target_dir = "./target_repo"
    
    chunks = chunker.process_repository(repo_url, target_dir)

    # remove chunks related to translation pages
    filtered_chunks = chunker.filter_chunks(chunks, '.ja.')
    logging.info(f"Number of kept docs: {len(filtered_chunks)}.")

    # Print example chunk with metadata
    for chunk in filtered_chunks[0:1]:
        logging.info("-"*24)
        print("Chunk content:", chunk.page_content[:100])
        print("Metadata:", chunk.metadata)
        logging.info("-"*24)

    
    # Export to CSV
    csv_path = "./data_folder/documents.csv"
    export_docs_to_csv(
        docs=filtered_chunks,
        output_path=csv_path,
        flatten_metadata=True,
        max_content_length=10_000
    )
    
    # Read back from CSV
    loaded_docs = read_docs_from_csv(csv_path, flatten_metadata=True)
    logging.info(f"Type of loaded_docs: {type(loaded_docs)}")
