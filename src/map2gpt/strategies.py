
import openai

import asyncio

import PyPDF2
import numpy as np

import operator as op

from map2gpt.model import Role, Message
from tiktoken import Encoding

from tenacity import retry, stop_after_attempt, wait_exponential

from tqdm import tqdm

from math import ceil 
from map2gpt.log import logger

from typing import List, Tuple, Dict, Optional, Any 

from urllib.parse import urlsplit, unquote

from sentence_transformers import SentenceTransformer
from typing import Awaitable, Callable, List, Optional, Tuple, Union

def load_transformer_model(model_name:str, cache_folder:str) -> SentenceTransformer:
    return SentenceTransformer(model_name, cache_folder=cache_folder)

def get_embedding(texts:str, model: SentenceTransformer, device:str) -> np.ndarray:
    return model.encode(texts, device=device)

def parse_url(url):
    pieces = urlsplit(url)
    netloc, path = op.attrgetter('netloc', 'path')(pieces)
    title = unquote(path).split('/')[-1].replace('_', ' ')
    language_code = netloc[:2]
    return title, language_code

def find_top_k(query_embedding:np.ndarray, embeddings:np.ndarray, k:int=5) -> List[Tuple[float, int]]:
    scores = query_embedding @ embeddings.T
    query_norm = np.linalg.norm(query_embedding)
    row_norms = np.linalg.norm(embeddings, axis=1)
    scaled_scores = scores / (query_norm * row_norms + 1e-8)
    indices = np.arange(len(embeddings))
    top_k_scores_indices = sorted(list(zip(scaled_scores, indices)), key=op.itemgetter(0), reverse=True)[:k]
    return top_k_scores_indices 

def convert_pdf2text( pdf_file:bytes):
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

def split_into_chunks(text_sentences:List[str], tokenizer:Encoding, chunk_size:int=128) -> List[str]:
    chunks_acc:List[str] = []
    tokens_acc:List[int] = []
    for sentence in text_sentences:
        tokens = tokenizer.encode(sentence)
        if len(tokens_acc) + len(tokens) > chunk_size:
            chunks_acc.append(tokenizer.decode(tokens_acc))
            tokens_acc = []
        else:
            tokens_acc.extend(tokens)
        
    return chunks_acc

def chunk_embeddings(transformer_model:SentenceTransformer, chunks:List[str], batch_size:int=8, device:str='cpu') -> np.ndarray:
    nb_chukns = len(chunks)
    nb_batches = ceil(nb_chukns / batch_size)
    embeddings_accumulator:List[List[float]] = []
    logger.info(f'nb_chunks: {nb_chukns} => nb_batches: {nb_batches} (batch_size: {batch_size}')
    for partition in np.array_split(chunks, nb_batches):
        partition_embeddings = transformer_model.encode(partition, show_progress_bar=True, device=device)
        embeddings_accumulator.append(partition_embeddings)

    stacked_embeddings = np.vstack(embeddings_accumulator)
    return stacked_embeddings

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def gpt3_question_answering(name:str, description:str, corpus_context:str, query:str) -> Optional[str]:
    completion_response = await openai.ChatCompletion.acreate(
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
async def gpt3_questions_generation(corpus_context:str) -> Optional[str]:
    completion_response = await openai.ChatCompletion.acreate(
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



async def gpt3_call(question_answering_awaitable:Awaitable[str], question_generation_awaitable:Awaitable[str]):
    return await asyncio.gather(
        question_answering_awaitable, 
        question_generation_awaitable,
        return_exceptions=True
    )
