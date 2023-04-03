

from map2gpt.runner import GPTRunner

from map2gpt.telegram_bot import GPTChatBOT

def deploy_on_telegram(telegram_token:str, runner:GPTRunner, name:str, description:str, top_k:int, source_k:int):
    with GPTChatBOT(token=telegram_token, chatbot_id='map2gpt', runner=runner, name=name, description=description, top_k=top_k, source_k=source_k) as bot:
        bot.listen()