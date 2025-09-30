"""
HTML Content Extractor
Removes all HTML tags and extracts clean text content from an HTML file.
"""

from bs4 import BeautifulSoup
import sys

def clean_html(input_file, output_file=None):
    """
    Extract text content from HTML file, removing all tags.

    Args:
        input_file: Path to input HTML file
        output_file: Path to output file (default: input_file with .txt extension)
    """
    # Read the HTML file
    with open(input_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Parse HTML and extract text
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text content
    text = soup.get_text()

    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    # Determine output file
    if output_file is None:
        output_file = input_file.rsplit('.', 1)[0] + '_cleaned.txt'

    # Write cleaned content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Cleaned content saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_html.py <input_html_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    clean_html(input_file, output_file)