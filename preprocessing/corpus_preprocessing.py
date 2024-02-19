import os
import argparse
from pdf_utils_package import pdfUtils

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract sentences from PDF files.')
    parser.add_argument('PDF_DIRECTORY', help='Directory containing PDF files')
    parser.add_argument('OUTPUT_PATH', help='Output JSONL file path')
    parser.add_argument('--lemmatizer', action='store_true', help='Disable lemmatization (default: False)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    pdf_paths = [os.path.join(args.PDF_DIRECTORY, file) for file in os.listdir(args.PDF_DIRECTORY) if file.lower().endswith(".pdf")]
    sentence_dataset = pdfUtils.create_sentence_dataset(pdf_paths, lemmatizer=args.lemmatizer)
    pdfUtils.write_jsonl(args.OUTPUT_PATH, sentence_dataset)

if __name__ == '__main__':
    main()

# Command line: 
# python script_name.py path/to/pdf_directory output_path.jsonl --lemmatizer