import re 
import openai

import PyPDF2
import numpy as np

import httpx 

import operator as op
import wikipedia 

from io import BytesIO

from map2gpt.model import Role, Message
from tiktoken import Encoding

from tenacity import retry, stop_after_attempt, wait_exponential

from math import ceil 
from map2gpt.log import logger

from typing import List, Tuple, Dict, Optional, Any 

from urllib.parse import urlsplit, unquote

from pytube import YouTube
from xml.etree import ElementTree
from html import unescape

from sentence_transformers import SentenceTransformer
from typing import Awaitable, Callable, List, Optional, Tuple, Union

from pydub import AudioSegment

def convert_ogg_sound2mp3_sound(ogg_bytes:BytesIO) -> BytesIO:
    ogg_audio = AudioSegment.from_file(ogg_bytes, format="ogg")
    mp3_bytes = BytesIO()
    ogg_audio.export(mp3_bytes, format="mp3")
    mp3_bytes.seek(0)
    return mp3_bytes

def load_transformer_model(model_name:str, cache_folder:str) -> SentenceTransformer:
    return SentenceTransformer(model_name, cache_folder=cache_folder)

def xml_caption2txt(xml_captions:str):
    sentences = []
    root = ElementTree.fromstring(xml_captions)
    for child in list(root.findall('body/p')):
        text = ''.join(child.itertext()).strip()
        if not text:
            continue
        caption = unescape(text.replace("\n", " ").replace("  ", " "),)
        sentences.append(caption)
    return sentences

def convert_text_into_sentences(text:str) -> Optional[List[str]]:
    try:
        sentence_regex = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        sentences = sentence_regex.split(text)
        return sentences
    except Exception as e:
        logger.error(f'Error while converting text into sentences: {e}')
        return None

def convert_wikipedia_page2text(url:str) -> Optional[List[str]]:
    pieces = urlsplit(url)
    netloc, path = op.attrgetter('netloc', 'path')(pieces)
    _, label, _ = netloc.split('.')
    if label == 'wikipedia':
        title = unquote(path).split('/')[-1].replace('_', ' ')
        language_code = netloc[:2]
        wikipedia.set_lang(language_code)
        page = wikipedia.page(title)
        return page.content.split('\n')
    return None 

def convert_youtube_link2text(url:str, language:str) -> Optional[List[str]]:
    pieces = urlsplit(url)
    netloc = op.attrgetter('netloc')(pieces)
    _, label, _ = netloc.split('.')
    if label == 'youtube':
        yt = YouTube(url)
        yt_captions = yt.captions
        captions = [ caption for caption in yt_captions if language in caption.name]
        if len(captions) == 0:
            logger.warning(f'no {language} caption were found for this youtube video {url}')
            return None  
        xml_captions = captions[0].xml_captions
        sentences = xml_caption2txt(xml_captions)
        return sentences
    return None

def convert_pdf_file2text(pdf_file:bytes) -> Optional[List[str]]:
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        nb_pages = len(reader.pages)
        logger.info(f'pdf_file => nb_pages: {nb_pages}')

        text_sentences = []
        for page_num in range(nb_pages):   
            page = reader.pages[page_num]
            page_content = page.extract_text()
            sentences = page_content.split('\n')
            text_sentences.extend(sentences)
        return text_sentences 
    except Exception as e:
        logger.error(f'Error while converting pdf file into text: {e}')
        return None

def merge_sentences_into_chunks(text_sentences:List[str], tokenizer:Encoding, chunk_size:int=128) -> List[str]:
    chunks_acc:List[str] = []
    tokens_acc:List[int] = []
    for sentence in text_sentences:
        tokens = tokenizer.encode(sentence)
        if len(tokens_acc) + len(tokens) > chunk_size:
            chunks_acc.append(tokenizer.decode(tokens_acc))
            tokens_acc = []
        else:
            tokens_acc.extend(tokens)
    if len(tokens_acc) > 0:
        chunks_acc.append(tokenizer.decode(tokens_acc))
    return chunks_acc

def chunk_embeddings(transformer_model:SentenceTransformer, chunks:List[str], batch_size:int=8, device:str='cpu') -> np.ndarray:
    nb_chukns = len(chunks)
    nb_batches = ceil(nb_chukns / batch_size)
    embeddings_accumulator:List[List[float]] = []
    logger.info(f'nb_chunks: {nb_chukns} => nb_batches: {nb_batches} | batch_size: {batch_size}')
    for partition in np.array_split(chunks, nb_batches):
        partition_embeddings = transformer_model.encode(partition, show_progress_bar=True, device=device)
        embeddings_accumulator.append(partition_embeddings)

    stacked_embeddings = np.vstack(embeddings_accumulator)
    return stacked_embeddings

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def gpt3_question_answering(name:str, description:str, corpus_context:str, query:str) -> Optional[str]:
    logger.info(f'gpt3_question_answering: {query}')
    completion_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            Message(
                role=Role.SYSTEM,
                content=f"""
                    ROLE : 
                        - Tu es l'assistant {name}. 
                        - Tu dois aider les utilisateur a trouver des réponses en analysant les documents fournis.
                        - Voici une description du service que tu offres : {description}
                    
                    FONCTIONNEMENT:
                       - Voici une suite de documents:
                       \"""
                          {corpus_context}
                       \"""
                       - Ces documents ont été selectionnés par une méthode de semantic search en analysant le message de l'utilisateur.
                       - Verifie si le message de l'utilisateur est en rapport avec les documents.
                          - Si oui Alors :
                              - Essaie de trouver une réponse à sa question dans les documents fournis.
                              - La réponse doit se baser sur les documents fournis.
                          - Si non, alors:
                              - Regarde si cest une demande d'information relative a ton service. 
                                - Si oui, réponds à cette demande d'information
                                - Si non, réponds que tu ne peux pas répondre à cette question.
                    ATTENTION:
                       - Tu n'as pas le droit de sortir du contexte de ta description et des documents fournis.
                       - Sois gentil avec l'utilisateur.
                       - Fais en sorte qu'il ait l'impression d'avoir une conversation avec un humain.
                """
            ).dict(),
            Message(
                role=Role.USER, 
                content=f"""
                    message: {query}
                """
            ).dict()
        ]
    )
    return completion_response['choices'][0]['message']['content']

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def gpt3_questions_generation(corpus_context:str) -> Optional[str]:
    logger.info(f'gpt3_questions_generation')
    completion_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            Message(
                role=Role.SYSTEM,
                content=f"""
                    ROLE: Tu dois générer des questions pertinentes en analysant les documents fournis.
                    FONCTIONNEMENT:
                        - Les documetns sont une suite de paragraphes selectionnés par une méthode de semantic search en analysant le message de l'utilisateur.
                        - Tu dois analyser le contenu de ces paragraphes et générer des questions pertinentes.
                        - Les questions doivent se baser sur les documents fournis.
                        - La réponse à ces questions doit se trouver dans les documents fournis.
                    ATTENTION:
                       - Tu n'as pas le droit de générer des questions dont les réponses ne sont pas dans les documents fournis.
                       - Tu ne dois pas répondre aux questions générées.
                       - Tu dois juste créer des questions!!!
                """
            ).dict(),
            Message(
                role=Role.USER, 
                content=f"""
                    DOCUMENTS: {corpus_context}
                    Génère moi seulement les questions!
                """
            ).dict()
        ]
    )

    return completion_response['choices'][0]['message']['content']

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def make_transcription(output:bytes, openai_api_key:str) -> Optional[str]:
    try:
        headers = {
            'Authorization': f'Bearer {openai_api_key}'
        }
        async with httpx.AsyncClient(headers=headers) as client:
            response = await client.post(
                url='https://api.openai.com/v1/audio/transcriptions',
                files={
                    'file': ('audio.mp3', output)
                },
                data={'model': 'whisper-1'},
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                text = data['text']
                return text  
            
            logger.warning(response.status_code)
            logger.warning(response.content)
    except httpx.TimeoutException as e:  
        logger.error(e)
    except Exception as e:
        logger.error(e)
    
    return None 

