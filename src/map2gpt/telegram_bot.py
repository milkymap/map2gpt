import re 

from io import BytesIO
import subprocess

import numpy as np 
from os import path 

import logging 

import numpy as np 

import asyncio 

from typing import List, Optional
from glob import glob 

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    filters, 
    Application,
    ContextTypes,
    CommandHandler,
    ConversationHandler,
    MessageHandler
)

from map2gpt.model  import Message, Role
from map2gpt.runner import GPTRunner
from map2gpt.strategies import convert_ogg_sound2mp3_sound, make_transcription, gpt3_questions_generation
import logging 

logging.basicConfig(
    format='%(asctime)s : (%(name)s) | %(filename)s -- %(lineno)3d -- %(levelname)7s -- %(message)s',
    level=logging.INFO 
)

logger = logging.getLogger(name='TelegramBot')

class GPTChatBOT:
    def __init__(self, token:str, chatbot_id:str, runner:GPTRunner, name:str, description:str, top_k:int, source_k:int):
        self.token = token 
        self.name = name 
        self.description = description
        self.chatbot_id = chatbot_id
        self.runner = runner
        self.top_k = top_k
        self.source_k = source_k
        
        self.app = Application.builder().token(token).build()
    
    async def start(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        user = update.message.from_user
        logger.debug(f'{user.first_name} is connected')
        await update.message.reply_text(text=f'Hello {user.first_name}, je suis le chatbot {self.name}. Voici la description de mon domaine de compétence : {self.description}')
        return 0 
    
    async def generate_questions(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        text = update.message.text.replace('/generate', '')
        if len(text) == 0:
            await update.message.reply_text(text='Veuillez entrer un texte pour générer des questions. Exemple : /generate les causes de la pandémie!')
            return 0
        logger.info(f'generating questions from {text}')
        messages = self.runner.generate_questions(query=text, collection_name=self.name, top_k=self.top_k)
        logger.info(f'generated questions : {messages}')
        if messages is not None:
           questions = messages.split('\n')
           questions = [ [question] for question in questions]
           await update.message.reply_text(text='Voici les questions générées :', reply_markup=ReplyKeyboardMarkup(questions, one_time_keyboard=True))
        else: 
            await update.message.reply_text(text='Une erreur est survenue durant la génération des questions, merci de refaire votre demande')
        return 0
    
    async def chatting(self, update:Update, context:ContextTypes.DEFAULT_TYPE):        
        text = update.message.text
        voice = update.message.voice

        if text is None and voice is not None:
            binarystream = await update.message.voice.get_file()
            bytearray = await binarystream.download_as_bytearray()
            logger.info('transcription en cours')
            mp3_binarystream = convert_ogg_sound2mp3_sound(BytesIO(bytearray))
            if mp3_binarystream is None:
                await update.message.reply_text('Une erreur est survenue durant la transcription, merci de refaire votre demande')
                return 0

            text = await make_transcription(mp3_binarystream.read(), self.runner.openai_api_key)
            logger.info(f'transcription : {text}')

        if text is None:
            await update.message.reply_text('Une erreur est survenue durant la transcription, merci de refaire votre demande')
            return 0 
        
        
        completion_response = self.runner.query_index(
            query=text, 
            collection_name=self.name, 
            top_k=self.top_k, 
            source_k=self.source_k, 
            description=self.description
        )

        if completion_response is not None:
            await update.message.reply_text(
                    text=completion_response
            )
            return 0

        await update.message.reply_text('Une erreur interne est survenue, merci de refaire votre demande')
        return 0 

    async def stop(self, update:Update, context:ContextTypes.DEFAULT_TYPE):
        user = update.message.from_user
        await update.message.reply_text(
            text=f"""
                Merci de votre visite {user.first_name} et n'hésitez pas à revenir si vous avez d'autres questions ou besoins d'assistance.À bientôt !
            """
        )
        return ConversationHandler.END

    def listen(self):
        self.app.run_polling()

    def __enter__(self):
        try:  
            handler = ConversationHandler(
                entry_points=[CommandHandler('start', self.start)],
                states={
                    0: [CommandHandler('stop', self.stop), CommandHandler('generate', self.generate_questions), MessageHandler(filters.TEXT|filters.VOICE, self.chatting)],
                },
                fallbacks=[CommandHandler('stop', self.stop)]
            )
            self.app.add_handler(handler)
        except Exception as e:
            logger.error(e)
        return self 

    def __exit__(self, exc, val, traceback):
        if exc is not None:
            logger.exception(traceback)
        logger.debug('GPTRunner ... shutdown')

