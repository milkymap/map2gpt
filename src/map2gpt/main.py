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

from map2gpt.runner import GPTRunner
from map2gpt.log import logger 

@click.group(chain=False, invoke_without_command=True)
@click.option('--openai_api_key', type=str, required=True, envvar='OPENAI_API_KEY')
@click.option('--cache_folder', type=str, required=True, envvar='TRANSFORMERS_CACHE')
@click.option('--transformer_model_name', type=str, default='Sahajtomar/french-semantic')
@click.pass_context
def group(ctx:click.core.Context, openai_api_key:str, cache_folder:str, transformer_model_name:str):
    ctx.ensure_object(dict)
    ctx.obj['openai_api_key'] = openai_api_key
    ctx.obj['transformer_model_name'] = transformer_model_name

    ctx.obj['cache_folder'] = cache_folder
    ctx.obj['tokenizer'] = tiktoken.encoding_for_model('gpt-3.5-turbo')

    device = th.device('cpu' if not th.cuda.is_available() else 'cuda')
    ctx.obj['device'] = device
    
    subcommand = ctx.invoked_subcommand
    if subcommand is not None:
        logger.debug(f"Invoked subcommand: {subcommand}")

@group.command()
@click.option('--path2pdf_file', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--path2extracted_features', type=click.Path(exists=False, dir_okay=False), required=True)
@click.option('--chunk_size', type=int, default=128, help='chunk size for the tokenizer')
@click.option('--batch_size', type=int, default=8, help='batch size for the cohere api')
@click.option('--name', type=str, required=True)
@click.option('--description', type=str, required=True)
@click.pass_context
def build_index_from_pdf(ctx:click.core.Context, path2pdf_file:str, path2extracted_features:str, chunk_size:int, batch_size:int, name:str, description:str):
    runner = GPTRunner(
        device=ctx.obj['device'],
        cache_folder=ctx.obj['cache_folder'], 
        openai_api_key=ctx.obj['openai_api_key'], 
        transformer_model_name=ctx.obj['transformer_model_name'], 
    )
    
    extracted_features = runner.build_index_from_pdf(path2pdf_file=path2pdf_file, chunk_size=chunk_size, batch_size=batch_size, name=name, description=description)
    runner.save_index(extracted_features=extracted_features, path2file=path2extracted_features)

@group.command()
@click.option('--wikipedia_url', type=str, help='wikipedia valid url', required=True)
@click.option('--path2extracted_features', type=click.Path(exists=False, dir_okay=False), required=True)
@click.option('--chunk_size', type=int, default=128, help='chunk size for the tokenizer')
@click.option('--batch_size', type=int, default=8, help='batch size for the cohere api')
@click.option('--name', type=str, required=True)
@click.option('--description', type=str, required=True)
@click.pass_context
def build_index_from_wikipedia(ctx:click.core.Context, wikipedia_url:str, path2extracted_features:str, chunk_size:int, batch_size:int, name:str, description:str):
    runner = GPTRunner(
        device=ctx.obj['device'],
        cache_folder=ctx.obj['cache_folder'], 
        openai_api_key=ctx.obj['openai_api_key'], 
        transformer_model_name=ctx.obj['transformer_model_name'], 
    )
    
    extracted_features = runner.build_index_from_wikipedia(wikipedia_url=wikipedia_url, chunk_size=chunk_size, batch_size=batch_size, name=name, description=description)
    runner.save_index(extracted_features=extracted_features, path2file=path2extracted_features)

@group.command()
@click.option('--path2extracted_features', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--top_k', type=int, default=7)
@click.option('--source_k', type=int, default=3)
@click.pass_context
def explore_index(ctx:click.core.Context, path2extracted_features:str, top_k:int, source_k:int):
    console = Console()
    openai.api_key = ctx.obj['openai_api_key']

    runner = GPTRunner(
        device=ctx.obj['device'],
        cache_folder=ctx.obj['cache_folder'], 
        openai_api_key=ctx.obj['openai_api_key'], 
        transformer_model_name=ctx.obj['transformer_model_name'], 
    )

    extracted_features = runner.load_index(path2extracted_features)    
    logger.info(f"Loaded extracted features from: {path2extracted_features}")

    keep_looping = True
    while keep_looping:
        try:
            query = input('USER: ')
            if query == 'exit':
                keep_looping = False
            else:
                
                index_response = runner.query_index(query=query, extracted_features=extracted_features, top_k=top_k, source_k=source_k)
                if index_response is not None:
                    ASSISTANT = index_response.answer
                    QUESTIONS = index_response.questions
                    SOURCE_CHUNKS = '\n\n'.join(index_response.source_chunks)
                    console.print(f"ASSISTANT: {ASSISTANT}", style="bold green")
                    console.print(f"QUESTIONS:\n{QUESTIONS}", style="bold yellow")
                    console.print(f"SOURCE_CHUNKS:\n{SOURCE_CHUNKS}", style="bold blue")
        except KeyboardInterrupt:
            keep_looping = False
        except Exception:
            logger.exception("Exception occurred")
            keep_looping = False

if __name__ == '__main__':
    group(obj={})