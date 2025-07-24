import pandas as pd
import numpy as np
import urllib.request
import xml.etree.ElementTree as ET
from PyPDF2 import PdfReader
import requests
from io import BytesIO
import re
import json
from datetime import datetime

def get_arxiv_paper_content(search_query="electron", max_results=1):
    """
    Fetch arXiv papers and extract their main content
    """
    # Get paper metadata
    url = f'http://export.arxiv.org/api/query?search_query=all:{search_query}&start=0&max_results={max_results}'
    
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read().decode('utf-8')
        
        # Parse XML
        root = ET.fromstring(xml_data)
        
        # Define namespaces
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        papers = []
        
        for entry in root.findall('atom:entry', namespaces):
            paper_info = {}
            
            # Get basic info
            title = entry.find('atom:title', namespaces)
            paper_info['name'] = title.text.strip() if title is not None else "No title"
            
            # Get PDF link
            links = entry.findall('atom:link', namespaces)
            pdf_url = None
            for link in links:
                if link.get('type') == 'application/pdf':
                    pdf_url = link.get('href')
                    break
            
            # Extract content if PDF is available
            if pdf_url:
                try:
                    content = extract_pdf_content(pdf_url)
                    paper_info['content'] = content
                except Exception as e:
                    print(f"Error extracting content from {pdf_url}: {e}")
                    paper_info['content'] = ""
            else:
                paper_info['content'] = ""
            
            papers.append(paper_info)
        
        return papers
    
    except Exception as e:
        print(f"Error fetching papers: {e}")
        return []

def extract_pdf_content(pdf_url):
    """
    Extract text content from PDF URL
    """
    try:
        # Add .pdf extension if not present
        if not pdf_url.endswith('.pdf'):
            pdf_url = pdf_url.replace('/abs/', '/pdf/') + '.pdf'
        
        # Download PDF
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(pdf_url, headers=headers)
        response.raise_for_status()
        
        # Read PDF
        pdf_file = BytesIO(response.content)
        pdf_reader = PdfReader(pdf_file)
        
        # Extract text from all pages
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        # Clean and extract main content
        cleaned_content = clean_pdf_text(text_content)
        
        return cleaned_content
    
    except Exception as e:
        raise Exception(f"PDF extraction error: {e}")

def clean_pdf_text(text):
    """
    Basic cleaning of extracted PDF text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove page numbers (simple pattern)
    text = re.sub(r'\n\d+\s*\n', '\n', text)
    
    # Remove arXiv footer patterns
    text = re.sub(r'arXiv:\d+\.\d+v\d+\s*\[.*?\]\s*\d+\s*\w+\s*\d+', '', text)
    
    # Trim to reasonable length if too long (optional)
    # if len(text) > 50000:
    #     text = text[:50000] + "... (truncated)"
    
    return text.strip()

def save_to_json(data, filename=None):
    """
    Save data to JSON file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arxiv_papers_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Data saved to {filename}")
    return filename

# Usage example
if __name__ == "__main__":
    # Install required packages first:
    # pip install PyPDF2 requests
    
    print("Fetching arXiv papers...")
    papers = get_arxiv_paper_content("neuralnetworks", 10)  # Get 10 papers as example
    
    # Save to JSON (papers already contains only name and content)
    json_filename = save_to_json(papers, "arxiv_papers.json")
    print("done")