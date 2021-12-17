import json
from collections import Counter
from os import path
from pathlib import Path
from typing import Union

import arabic_reshaper
import hazm
import loguru as logger
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from src.Data import DATA_DIR
from wordcloud import WordCloud

"""Generating wordcloud from the exported json telegram chat 
   put a json file in src/data folder
"""
# chat_json = DATA_DIR / 'online.json'
# print(str(chat_json))
class ChatStatistics:
    def __init__(self, chat_json: Union[str, Path]):
     # Loading Chat data 
        with open(chat_json) as f:
            self.chat_data = json.load(f)
        logger.logger.info('Generating WordCloud from Json files...')

     # loading Stop words
        logger.logger.info("Loading Stopwords from fa_stop_words.txt...")
        fa_stop_words = open(DATA_DIR / 'fa_stop_words.txt').readlines()
        stopwords = list(map(str.strip, fa_stop_words))
        self.normalizer = hazm.Normalizer()
        self.stopwords = list(map(self.normalizer.normalize, fa_stop_words))    
        
   # Generating Data Content
    def wordcloud_generator(self, output_dir: Union[str, Path]):
        self.text_content = ''        
        for msg in self.chat_data['messages']:
            if type(msg['text']) is str:
                tokens = hazm.word_tokenize(msg['text'])
                tokens = list(filter(lambda item: item not in self.stopwords, tokens))
                self.text_content += f" {tokens}"
        
        # Normalizing Data Content
        logger.logger.info("Normalizing Chat messages...")
        self.text_content = self.normalizer.normalize(self.text_content)

        #Reshaping of Arabic and Persian Texts
        self.text_content = arabic_reshaper.reshape(self.text_content)
        # self.text_content = get_display(self.text_content)
        
        # generating, plotting the wordcloud
        # fp = DATA_DIR / 'fonts/BHoma.ttf'
        logger.logger.info('Generating WordCloud from Json files...')
        wordcloud = WordCloud(background_color='white',
                                max_font_size=80,
                                width=800, 
                                height=400,
                                font_path= str(Path(DATA_DIR / 'fonts/BHoma.ttf'))).generate(self.text_content)
        logger.logger.info(f'Saving WordCloud as a png file in {output_dir}...')
        wordcloud.to_file(output_dir / 'wc_output_TT_3.png')
        

if __name__ == '__main__':
    chat_stat = ChatStatistics(chat_json=DATA_DIR / 'result.json')
    chat_stat.wordcloud_generator(output_dir=DATA_DIR)
    print('Generating wordcloud is done...')
