"""from data.helpers.pymupdf_rag import to_markdown
print(to_markdown('2307.01361.pdf', pages=[4]))"""
"""from data.data_maker import process_file

process_file('2307.01361.pdf', '2307.01361.json')"""

"""import pdfplumber

with pdfplumber.open("2308.08384.pdf") as pdf:
    first_page = pdf.pages[10]
    print(first_page.chars[0])"""

"""from pypdf import PdfReader
reader = PdfReader("2308.08384.pdf")
page = reader.pages[10]
print(page.extract_text())
"""

from olmocr.prompts.anchor import _pdf_report

objs = _pdf_report('2308.08384.pdf', 11)
print('done')