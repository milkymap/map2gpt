import os 
import click 

import openai 
import wikipedia
import asyncio

import numpy as np 
import torch as th 

import pickle 

import tiktoken

from rich.console import Console

from map2gpt.log import logger 
from map2gpt.strategies import (
    convert_pdf2text, 
    split_into_chunks, 
    chunk_embeddings, 
    gpt3_question_answering, 
    gpt3_questions_generation, 
    find_top_k, 
    parse_url, 
    load_transformer_model, 
    get_embedding,
    gpt3_call
)

from typing import Optional, List, Dict, Any, Union

from map2gpt.model import IndexResponse, ExtractedFeatures

class GPTRunner:
    def __init__(self, openai_api_key:str, cache_folder:str, transformer_model_name:str, device:str='cpu', tokenizer:str='gpt-3.5-turbo'):
        self.openai_api_key = openai_api_key
        self.transformer_model_name = transformer_model_name
        self.cache_folder = cache_folder
        self.device = device
        self.tokenizer = tiktoken.encoding_for_model(tokenizer)

        self.transformer_model = load_transformer_model(
            self.transformer_model_name,
            self.cache_folder
        )

        openai.api_key = self.openai_api_key


    def build_index_from_wikipedia(self, wikipedia_url:str, chunk_size:int, batch_size:int, name:str, description:str) -> ExtractedFeatures:
        title, language_code = parse_url(wikipedia_url)
        wikipedia.set_lang(language_code)
        page = wikipedia.page(title)

        extracted_sentences = page.content.split('\n')
        logger.info('text was extracted from wikipedia')
        chunks = split_into_chunks(extracted_sentences, tokenizer=self.tokenizer, chunk_size=chunk_size)
        logger.info(f'nb_sentences : {len(extracted_sentences)} | nb_chunks : {len(chunks)}')

        transformer_model = load_transformer_model(self.transformer_model_name, cache_folder=self.cache_folder)
        embeddings = chunk_embeddings(
            transformer_model=transformer_model,
            chunks=chunks,  
            batch_size=batch_size,
            device=self.device
        )

        extracted_features = {'chunks': chunks, 'embeddings': embeddings, 'name':name, 'description':'description'}
        return ExtractedFeatures(**extracted_features)

    def build_index_from_pdf(self, path2pdf_file:str, chunk_size:int, batch_size:int, name:str, description:str) -> ExtractedFeatures:
        extracted_sentences = convert_pdf2text(path2pdf_file)
        logger.info('text was extracted from pdf')
        chunks = split_into_chunks(extracted_sentences, tokenizer=self.tokenizer, chunk_size=chunk_size)
        logger.info(f'nb_sentences : {len(extracted_sentences)} | nb_chunks : {len(chunks)}')

        transformer_model = load_transformer_model(self.transformer_model_name, cache_folder=self.cache_folder)
        embeddings = chunk_embeddings(
            transformer_model=transformer_model,
            chunks=chunks,  
            batch_size=batch_size,
            device=self.device
        )
        extracted_features = {'chunks': chunks, 'embeddings': embeddings, 'name':name, 'description':'description'}
        return ExtractedFeatures(**extracted_features)

    def save_index(self, extracted_features:ExtractedFeatures, path2file:str) -> None:
        with open(path2file, 'wb') as f:
            pickle.dump(extracted_features, f)
        logger.info(f'index was saved at {path2file}')
    
    def load_index(self, path2file:str) -> ExtractedFeatures:
        with open(path2file, 'rb') as f:
            extracted_features = pickle.load(f)
        logger.info(f'index was loaded from {path2file}')
        return extracted_features

    def query_index(self, query:str, extracted_features:ExtractedFeatures, top_k:int=7, source_k:int=2) -> Optional[IndexResponse]:
        assert source_k <= top_k, 'source_k must be less than or equal to top_k'
        assert top_k <= len(extracted_features.chunks), 'top_k must be less than or equal to the number of chunks'

        query_embedding = self.transformer_model.encode(query, device=self.device, show_progress_bar=False)
        selected_scores_indices = find_top_k(query_embedding, extracted_features.embeddings, k=top_k)
        corpus_embeddings = [extracted_features.embeddings[index] for _, index in selected_scores_indices]
        corpus_embeddings = np.vstack(corpus_embeddings)

        chunks_acc = []
        for _, index in selected_scores_indices:
            chunk = extracted_features.chunks[index]
            chunks_acc.append(chunk)
        corpus_context = '\n'.join(chunks_acc)
        
        try:

            logger.debug('Searching...')
            answer, questions = asyncio.run(main=gpt3_call(
                question_answering_awaitable=gpt3_question_answering(
                    name=extracted_features.name,
                    query=query,
                    description=extracted_features.description,
                    corpus_context=corpus_context
                ),
                question_generation_awaitable=gpt3_questions_generation(corpus_context=corpus_context)
            ))
            
            if isinstance(answer, str) and isinstance(questions, str): 
                response_embedding = self.transformer_model.encode(answer, device=self.device, show_progress_bar=False)
                source_scores_indices = find_top_k(response_embedding, corpus_embeddings, k=source_k)
                source_chunks = [chunks_acc[index] for _, index in source_scores_indices]
                response = {
                    'answer': answer,
                    'questions': questions,
                    'source_chunks': source_chunks
                }
                return IndexResponse(**response)
                
            raise Exception('GPT-3 did not return a valid answer or questions')
        except Exception as e:
            logger.exception("Exception occurred")
            logger.error(f"Exception: {e}")

        return None 
    
