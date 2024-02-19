# !pip install num2words
# !pip install contractions
# !pip install emoji
# !pip install pyspellchecker
from tqdm.auto import tqdm
import fitz
import re
from datetime import datetime
import spacy
from num2words import num2words
import contractions
import emoji
from spellchecker import SpellChecker
from unidecode import unidecode
import jsonlines

nlp = spacy.load("en_core_web_lg")
spell = SpellChecker()

def pdf_creation_date_to_unix_timestamp(pdf_date_string):
    try:
        # Extraemos fecha y tiempo de creación del PDF
        pdf_date_string = pdf_date_string.replace("D:", "").replace("'", "")
        pdf_date_obj = datetime.strptime(pdf_date_string, "%Y%m%d%H%M%S%z")

        # Convertimos de datetime object a Unix timestamp
        unix_timestamp = int(pdf_date_obj.timestamp())
        return unix_timestamp
    except Exception as e:
        print(f"Error: {e}")

def word_lemmatizer(tokens):
    lemmas = []
    for token in tokens:
        doc = nlp(token)
        lemmas.append(doc[0].lemma_)
    return lemmas

def convert_numbers_to_words(text):
    words = []
    for word in text.split():
        if word.isnumeric():
            words.append(num2words(word))
        else:
            words.append(word)
    return ' '.join(words)

def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text

def convert_emojis_to_words(text):
    clean_text = ''
    for char in text:
        if char not in emoji.EMOJI_DATA:
            clean_text += char
    return clean_text


def convert_accented_to_ascii(text):
    return unidecode(text)


def correct_spelling(sentence):
    corrected_sentence = []
    words = sentence.split()
    for word in words:
        corrected_word = spell.correction(word)
        corrected_sentence.append(corrected_word)
    return " ".join(corrected_sentence)


# Regex pattern para eliminar etiquetas HTML
html_tags_pattern = re.compile(r'<.*?>')
# Regex pattern para URLs
url_pattern = re.compile(r'https?://\S+|www\.\S+')
# Regex pattern para separar strings con ".", "!", "?",
# las commas: "," y las "apostrophes: "'" se quedan puesto que son útiles para NER
sentence_pattern = re.compile(r'[.!?](?:\s|\Z)')
# Regex pattern para separar strings como "Report2018Insight" o "ReportThe"
pattern = re.compile(r'(?<=[a-z])(?=[A-Z0-9()])|(?<=[0-9()])(?=[A-Za-z])|(?<=[A-Za-z])(?=[0-9()])|(?<=[a-zA-Z])(?=[A-Z][a-z])|(?<=[a-zA-Z])(?=[!@#$%^&*()_+{}|:;"<>,.?/~`])')
# Regex pattern para números
number_pattern = re.compile(r'\b\d+\b')
# Regex pattern para eliminar characteres singulares
single_character_pattern = re.compile(r'(?<!\S).(?!\S)\s*')
special_characters_pattern = re.compile(r'[!¡“£¢∞§§¶•ªº≠«π^…µλπøı¨@∑€®†¥ß∂ƒ©@#$%^&*+_{}()|:;"<>,?/~`]')

def extract_sentences_from_pdf(pdf_path, lemmatizer=True):
    try:
      doc = fitz.open(pdf_path)
      metadata = doc.metadata
      utc = pdf_creation_date_to_unix_timestamp(metadata['creationDate'])
      sentences = []
      
      for page_num in tqdm(range(doc.page_count), desc="Processing pages"):
          page = doc.load_page(page_num)
          text = page.get_text("text")
          text = text.replace('\n', ' ')
          text = text.replace('\r', ' ')
          
          if (len(text) < 5):
              continue
          
          split_text = re.split(sentence_pattern, text)
          
          for split_sentence in split_text:
              sentence = ' '.join(re.split(pattern, split_sentence))

              # Eliminar etiquetas HTML
              sentence = re.sub(html_tags_pattern, '', sentence)
              
              # Eliminar URLs
              sentence = re.sub(url_pattern, '', sentence)
              
              # Convertir números a palabras [LOS NÚMEROS NO SON MUY RELEVANTES EN NUESTRO CASO]
              # sen = convert_numbers_to_words(sen)
              
              # Eliminar números
              sentence = re.sub(number_pattern, '', sentence)
              
              # Checamos spelling
              # sen = correct_spelling(sen)
              
              # Expandimos contracciones
              sentence = expand_contractions(sentence)
              
              # Convertir caracteres acentuados a ASCII
              sentence = convert_accented_to_ascii(sentence)
    
              sentence = ' '.join(re.split(pattern, sentence))
              
              # Convertir emojis a palabras
              sentence = convert_emojis_to_words(sentence)
              
              if (lemmatizer):
                    # Lematización
                    sentence = word_lemmatizer(sentence.split())
                    sentence = ' '.join(sentence)
                    # Eliminar caracteres singulares
                    sentence = re.sub(single_character_pattern, '', sentence)

                    sentence = re.sub(special_characters_pattern, '', sentence)
                    sentence = sentence.replace('-', ' ')
                    sentences.append(sentence)
              else:
                  # Eliminar caracteres singulares
                  sentence = re.sub(single_character_pattern, '', sentence)

                  sentence = re.sub(special_characters_pattern, '', sentence)
                  sentence = sentence.replace('-', ' ')
                  sentences.append(sentence)
              
      print("Doc: ",
        pdf_path.split('/')[1],
        "\t| # pages: ",
        doc.page_count,
        "\t| # sentences: ", len(sentences))

      doc.close()
      # Filtramos strings vacios
      return list(filter(None, sentences)), utc

    except Exception as e:
       print(f"Error: {e}")


def create_sentence_dataset(pdf_paths, lemmatizer=True):
    sentences_objects = []

    for pdf_path in pdf_paths:
        sentences, utc = extract_sentences_from_pdf(pdf_path, lemmatizer=lemmatizer)
        for sentence in sentences:
          sentence_object = {
             "text": sentence.strip(),
             "utc": utc if utc else None
          }
          sentences_objects.append(sentence_object)
    return sentences_objects


def write_jsonl(file_path, data):
    with jsonlines.open(file_path, 'w') as writer:
        for item in data:
            writer.write(item)