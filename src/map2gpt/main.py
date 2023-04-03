import os 

import click 
import openai 

import tiktoken
import numpy as np 
import torch as th 

from os import path 
from glob import glob 

from qdrant_client import QdrantClient

from map2gpt.runner import GPTRunner
from map2gpt.log import logger 

from map2gpt.model import YouTubeTranscriotionLanguage
from map2gpt.deploy import deploy_on_telegram

from typing import List 

@click.group(chain=False, invoke_without_command=True)
@click.option('--openai_api_key', type=str, required=True, envvar='OPENAI_API_KEY')
@click.option('--transformers_cache', type=click.Path(exists=True, dir_okay=True), required=True, envvar='TRANSFORMERS_CACHE')
@click.option('--transformer_model_name', type=str, default='Sahajtomar/french_semantic')
@click.option('--qdrant_persistent_folder', type=click.Path(exists=True, dir_okay=True), required=True, envvar='QDRANT_PERSISTENT_FOLDER')
@click.pass_context
def group(ctx:click.core.Context, openai_api_key:str, transformers_cache:str,  transformer_model_name:str, qdrant_persistent_folder:str):
    ctx.ensure_object(dict)
    device = th.device('cpu' if not th.cuda.is_available() else 'cuda')
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

    ctx.obj['device'] = device
    ctx.obj['tokenizer'] = tokenizer 
    ctx.obj['openai_api_key'] = openai_api_key
    
    ctx.obj['transformers_cache'] = transformers_cache
    ctx.obj['transformer_model_name'] = transformer_model_name
    
    ctx.obj['qdrant'] = QdrantClient(path=qdrant_persistent_folder)

    subcommand = ctx.invoked_subcommand
    if subcommand is not None:
        logger.debug(f"Invoked subcommand: {subcommand}")

@group.command()
@click.option('--name', type=str, required=True)
@click.option('--chunk_size', type=int, default=128, help='chunk size for the tokenizer')
@click.option('--batch_size', type=int, default=8, help='batch size for the cohere api')
@click.option('--path2pdf_files', type=click.Path(exists=True, dir_okay=False), required=True, multiple=True)
@click.pass_context
def build_index_from_pdf_files(ctx:click.core.Context, name:str, chunk_size:int, batch_size:int, path2pdf_files:List[str]):
    runner = GPTRunner(
        qdrant=ctx.obj['qdrant'],
        device=ctx.obj['device'],
        openai_api_key=ctx.obj['openai_api_key'], 
        transformers_cache=ctx.obj['transformers_cache'], 
        transformer_model_name=ctx.obj['transformer_model_name'], 
    )
    
    knowledge_base = runner.build_index_from_pdf_files(path2pdf_files=path2pdf_files, chunk_size=chunk_size, batch_size=batch_size, name=name)
    runner.create_qdrant_index(knowledge_base=knowledge_base)

@group.command()
@click.option('--urls', type=str, help='wikipedia valid url', required=True, multiple=True)
@click.option('--name', type=str, required=True)
@click.option('--chunk_size', type=int, default=128, help='chunk size for the tokenizer')
@click.option('--batch_size', type=int, default=8, help='batch size for the cohere api')
@click.pass_context
def build_index_from_wikipedia_pages(ctx:click.core.Context, urls:List[str], name:str, chunk_size:int, batch_size:int):
    runner = GPTRunner(
        qdrant=ctx.obj['qdrant'],
        device=ctx.obj['device'],
        openai_api_key=ctx.obj['openai_api_key'],
        transformers_cache=ctx.obj['transformers_cache'],  
        transformer_model_name=ctx.obj['transformer_model_name'], 
    )
    
    knowledge_base = runner.build_index_from_wikipedia_pages(urls=urls, chunk_size=chunk_size, batch_size=batch_size, name=name)
    runner.create_qdrant_index(knowledge_base=knowledge_base)

@group.command()
@click.option('--urls', type=str, help='youtube links', required=True, multiple=True)
@click.option('--name', type=str, required=True)
@click.option('--chunk_size', type=int, default=128, help='chunk size for the tokenizer')
@click.option('--batch_size', type=int, default=8, help='batch size for the cohere api')
@click.option('--language', type=YouTubeTranscriotionLanguage, default=YouTubeTranscriotionLanguage.FR)
@click.pass_context
def build_index_from_youtube_links(ctx:click.core.Context, urls:List[str], name:str, chunk_size:int, batch_size:int, language:YouTubeTranscriotionLanguage):
    runner = GPTRunner(
        qdrant=ctx.obj['qdrant'],
        device=ctx.obj['device'],
        openai_api_key=ctx.obj['openai_api_key'],
        transformers_cache=ctx.obj['transformers_cache'],  
        transformer_model_name=ctx.obj['transformer_model_name'], 
    )
    
    knowledge_base = runner.build_index_from_youtube_links(urls=urls, chunk_size=chunk_size, batch_size=batch_size, name=name, language=language)
    runner.create_qdrant_index(knowledge_base=knowledge_base)

@group.command()
@click.option('--path2directory', type=click.Path(exists=True, dir_okay=True), help='text content', required=True)
@click.option('--name', type=str, required=True)
@click.option('--chunk_size', type=int, default=128, help='chunk size for the tokenizer')
@click.option('--batch_size', type=int, default=8, help='batch size for the cohere api')
@click.pass_context
def build_index_from_texts(ctx:click.core.Context, path2directory:List[str], name:str, chunk_size:int, batch_size:int):
    text_filepaths = sorted(glob(path.join(path2directory, '*.txt')))
    text_contents = []
    for text_filepath in text_filepaths[:30]:
        with open(text_filepath, 'r') as file_pointer:
            text_contents.append(file_pointer.read())

    runner = GPTRunner(
        qdrant=ctx.obj['qdrant'],
        device=ctx.obj['device'],
        openai_api_key=ctx.obj['openai_api_key'],
        transformers_cache=ctx.obj['transformers_cache'],  
        transformer_model_name=ctx.obj['transformer_model_name'], 
    )
    
    knowledge_base = runner.build_index_from_texts(texts=text_contents, chunk_size=chunk_size, batch_size=batch_size, name=name)
    runner.create_qdrant_index(knowledge_base=knowledge_base)

@group.command()
@click.option('--name', type=str, required=True)
@click.option('--query', type=str, required=True)
@click.option('--top_k', type=int, default=7)
@click.option('--source_k', type=int, default=3)
@click.option('--description', type=str, required=True)
@click.pass_context
def query_index(ctx:click.core.Context, name:str, query:str, top_k:int, source_k:int, description:str):
    openai.api_key = ctx.obj['openai_api_key']
    try:
        ctx.obj['qdrant'].get_collection(collection_name=name)
        
        runner = GPTRunner(
            qdrant=ctx.obj['qdrant'],
            device=ctx.obj['device'],
            transformers_cache=ctx.obj['transformers_cache'], 
            openai_api_key=ctx.obj['openai_api_key'], 
            transformer_model_name=ctx.obj['transformer_model_name'], 
        )

        completion_response = runner.query_index(query=query, collection_name=name, top_k=top_k, source_k=source_k, description=description)
        print(completion_response)
    except Exception as e:
        logger.exception(e)

@group.command()
@click.option('--name', type=str, required=True)
@click.option('--top_k', type=int, default=7)
@click.option('--source_k', type=int, default=3)
@click.option('--description', type=str, required=True)
@click.option('--telegram_token', type=str, required=True)
@click.pass_context
def deploy_index_on_telegram(ctx:click.core.Context, name:str, top_k:int, source_k:int, description:str, telegram_token:str):
    openai.api_key = ctx.obj['openai_api_key']

    try:
        ctx.obj['qdrant'].get_collection(collection_name=name)
        
        runner = GPTRunner(
            qdrant=ctx.obj['qdrant'],
            device=ctx.obj['device'],
            transformers_cache=ctx.obj['transformers_cache'], 
            openai_api_key=ctx.obj['openai_api_key'], 
            transformer_model_name=ctx.obj['transformer_model_name'], 
        )

        deploy_on_telegram(telegram_token=telegram_token, name=name, top_k=top_k, source_k=source_k, description=description, runner=runner)
    except Exception as e:
        logger.exception(e)

if __name__ == '__main__':
    group(obj={})