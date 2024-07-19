import re
from tqdm import tqdm
from itertools import combinations
from collections import Counter
import argparse

import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn

import spacy
import fasttext
import jamspell

import networkx as nx

class Preprocessor():
    def __init__(self):
        # Initialize Jamspell Corrector
        jamspell_corrector = jamspell.TSpellCorrector()
        jamspell_corrector.LoadLangModel('./utils/en.bin')
        self.corrector = jamspell_corrector.FixFragment

        # Dataset Configuration
        self.roblox1 = pd.read_csv('./../reviews/roblox1.csv', index_col = 0)
        self.roblox2 = pd.read_csv('./../reviews/roblox2.csv', index_col = 0)
        self.roblox3 = pd.read_csv('./../reviews/roblox3.csv', index_col = 0)
        self.roblox4 = pd.read_csv('./../reviews/roblox4.csv', index_col = 0)
        self.roblox5 = pd.read_csv('./../reviews/roblox5.csv', index_col = 0)
        self.zepeto = pd.read_csv('./../reviews/zepeto.csv', index_col = 0)

        self.lemmatizer = WordNetLemmatizer()
        self.stopword_list = stopwords.words('english')

        # Configure spaCy and FastText
        self.load_model = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])
        self.lang_detect_model = fasttext.load_model('./../utils/lid.176.bin')

    # Detect Language
    def detect_language(self, text):
        predictions = self.lang_detect_model.predict(text, k=1)
        lang = predictions[0][0].split("__label__")[1]
        return str(lang)

    # Remove special characters or some patterns
    def clean_str(self, text):
        pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'    # E-mail
        text = re.sub(pattern=pattern, repl='', string=text)
        
        pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'   # URL
        text = re.sub(pattern=pattern, repl='', string=text)
        
        text = text.replace('.', '. ')                                  # Add space after punctuations
        text = text.replace(',', ', ')
        text = text.replace('?', '? ')
        text = text.replace('!', '! ')
        
        pattern = '<[^>]*>'                                             # HTML
        text = re.sub(pattern=pattern, repl='', string=text)

        pattern = '[^\w\s]'                                             # special characters
        text = re.sub(pattern=pattern, repl='', string=text)

        return text

    # Text Normalization
    def text_normalize(self, text):
        text = text.replace(' s ', ' is ')
        text = text.replace('n t ', ' not ')
        text = text.replace(' dont ', ' do not ')
        text = text.replace(' hasnt ', ' has not ')
        text = text.replace(' wouldnt ', ' would not ')
        text = text.replace(' havent ', ' have not ')
        text = text.replace(' couldnt ', ' could not ')
        text = text.replace(' didnt ', ' did not ')
        text = text.replace(' shouldnt ', ' should not ')
        text = text.replace(' wo ', ' will ')
        text = text.replace(' d ', ' would ')
        text = text.replace(' ve ', ' have ')
        text = text.replace(' ive ', ' i have ')
        text = text.replace(' heve ', ' he have ')
        text = text.replace(' sheve ', ' she have ')
        text = text.replace(' weve ', ' we have ')
        text = text.replace(' theyve ', ' they have ')
        text = text.replace(' re ', ' are ')
        text = text.replace(' cant ', ' can not ')
        text = text.replace(' ca ', ' can ')
        text = text.replace(' s ', ' is ')
        text = text.replace(' & ', ' and ')
        text = text.replace(' m ', ' am ')
        text = text.replace(' alot ', ' lot ')
        text = text.replace(' id ', ' i would ')
        text = text.replace(' its ', ' it is ')

        return text

    def run(self, domain = 'roblox'):
        if domain == 'roblox':
            dfs = [self.roblox1, self.roblox2, self.roblox3, self.roblox4, self.roblox5]
        elif domain == 'zepeto':
            dfs = [self.zepeto]
        elif domain == 'all':
            dfs = [self.roblox1, self.roblox2, self.roblox3, self.roblox4, self.roblox5, self.zepeto]
        else:
            raise ValueError('The domain should be one of [roblox, zepeto, all]!')

        # Save original copy 
        for df in dfs:
            df['original'] = df['content'].copy()

        # Drop NA
        for index, df in enumerate(dfs):
            dfs[index] = df.dropna(subset = ['content']).reset_index(drop = True)

        # Filter Language
        for index, df in enumerate(dfs):
            df['language'] = df['content'].progress_apply(self.detect_language)
            dfs[index] = df[df['language'] == 'en']

        # Jamspell Correction
        for df in dfs:
            df['content'] = df['content'].progress_apply(self.corrector)

        # Lowercasing
        for df in dfs:
            df['content'] = df['content'].progress_apply(lambda x : x.lower())

        # Remove special characters
        for df in dfs:
            df['content'] = df['content'].progress_apply(self.clean_str)

        # Remove Non-alphabet words 
        for df in dfs:
            df['content'] = df['content'].progress_apply(lambda x : re.sub('[^a-zA-Z]+', ' ', x))

        # Text Normalization (ex: 've -> have)
        for df in dfs:
            df['content'] = df['content'].progress_apply(self.text_normalize)

        # Tokenization: NLTK word tokenizer 사용.
        for df in dfs:
            df['content'] = df['content'].progress_apply(word_tokenize)

        # Stopword removal: nltk.corpus
        for df in dfs:
            df['content'] = df['content'].progress_apply(lambda x : [word for word in x if word not in self.stopword_list])

        # Merge into string
        for df in dfs:
            df['content'] = df['content'].progress_apply(lambda x : ' '.join(x))

        # Lemmatization
        for df in dfs:
            df['document'] = df['content'].progress_apply(self.load_model)
        for df in dfs:
            df['lemmas'] = df['document'].progress_apply(lambda x : [token.lemma_ for token in x])
        for df in dfs:
            df['content'] = df['lemmas']

        # Extract verbs, nouns, adjectives only
        for df in dfs:
            df['content'] = df['content'].progress_apply(lambda x : [word[0] for word in pos_tag(x) if word[1] in ['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ', 'JJ','JJR','JJS']])

        # Leave words that have at least three letters in it.
        for df in dfs:
            df['content'] = df['content'].progress_apply(lambda x : [word for word in x if len(word) > 2])

        # Fixing several unfiltered typos
        convert_dict = {
            'robox': 'robux',
            'rubox': 'robux',
            'rubux': 'robux',
            'robuxs': 'robux',
            'robuck': 'robux',
            'robloxs': 'roblox',
            'glih': 'glitch',
            'glich': 'glitch',
            'glitche': 'glitch',
            'gliche': 'glitch',
            'glichy': 'glitchy',
            'tictok': 'tiktok',
            'freind': 'friend',
            'agian': 'again',
            'accountplaye': 'accountplayer',
            'reccomend': 'recommend',
            'lagg': 'lag',
            'laggs': 'lag',
            'uninstalle': 'uninstall',
            'frend': 'friend',
            'n00b': 'noob',
            'n0oB': 'noob',
            'newb': 'newbie',
            'nube': 'newbie',
            'customise': 'customize',
            'lagy': 'laggy',
            'pls': 'please',
            'gem': 'zem',
            'glitche': 'glitch',
            'uninstalle': 'uninstall',
            'zepeti': 'zepeto',
            'zepito': 'zepeto',
            'srceen': 'screen',
            'geams': 'game',
            'laggie': 'laggy',
            'robloxs': 'roblox',
            'scams': 'scam',
            'devs': 'dev',
            'robuxs': 'robux',
            'emojis': 'emoji',
            'zepetos': 'zepeto',
            'noises': 'noise'
        }

        for df in dfs:
            df['content'] = df['content'].progress_apply(lambda x : [convert_dict[word] if word in convert_dict.keys() else word for word in x])

        exception_words = ['roblox', 'robux', 'bloxburg', '​​wifi', 'tycoon', 'chromebook', 'brookhaven', 'minecraft', 'xbox', 'youtuber', 'fortnite', 'parkour', 'minigame', 'tik', 'gacha', 'ninja', 'pokemon', 'rebbeca', 'kitkat', 'teleport', 'xbox', 'tiktok', 'newbie', 'noob', 'iphone', 'bedwar', 'gacha', 'asap', 'granny', 'covid', 'adventure', 'zepeto', 'zem', 'tiktok', 'tok', 'vpn', 'kpop', 'instagram', 'spyware', 'emoji', 'gacha', 'dawson', 'git', 'gmail', 'asap', 'selfie', 'wacthing', 'cache', 'youtuber', 'screenshot', 'ios', 'ariana', 'afro', 'zepetogram', 'darkweb',] + list(convert_dict.values())

        for df in dfs:
            df['content'] = df['content'].progress_apply(lambda x : [word for word in x if len(wn.synsets(word)) != 0 or word in exception_words])

        # Leave only reviews that have at least 5 words in it.
        for df in dfs:
            df = df[df['content'].progress_apply(len) >= 5].reset_index(drop = True)

        # Return final results
        if domain == 'roblox':
            return pd.concat(dfs).reset_index(drop = True)
        elif domain == 'zepeto':
            return dfs[0].reset_index(drop = True)
        elif domain == 'all':
            #  리뷰 수 5개 미만인 리뷰 제거: 각 리스트 내 element 개수가 5 이상인 경우만 남김.
            roblox_df = roblox_df[roblox_df['content'].progress_apply(len) >= 5].reset_index(drop = True)
            zepeto_df = zepeto_df[zepeto_df['content'].progress_apply(len) >= 5].reset_index(drop = True)

            roblox_df = pd.concat(dfs[:5]).reset_index(drop = True)
            zepeto_df = dfs[5].reset_index(drop = True)

            return roblox_df, zepeto_df
        
class GraphAnalyzer():
    def __init__(self, df):
        self.df = df

        self.count = Counter(word for sublist in self.df['content'] for word in sublist)
        self.G = nx.Graph()

    def run(self):
        self.sorted_keys = sorted(self.count, key = self.count.get, reverse=True)
        self.sorted_values = [self.count[key] for key in self.sorted_keys]

        for i in range(len(self.sorted_keys)):
            self.G.add_node(self.sorted_keys[i],count=self.sorted_values[i])

        for sen in tqdm(self.df['content']):
            comb_two_ele = list(combinations(set(sen), 2))
            for two_ele in comb_two_ele:
                u = two_ele[0]
                v = two_ele[1]
                if u in self.sorted_keys and v in self.sorted_keys:
                    if (u, v) in self.G.edges():
                        self.G[u][v]['weight']+=1
                    else:
                        self.G.add_edge(u, v, weight=1)

        # Compute closeness centrality
        closeness_centrality = {}
        for node in tqdm(self.G.nodes(), desc='Closeness centrality 계산'):
            closeness_centrality[node] = nx.closeness_centrality(self.G, u = node, distance = 'weight')

        nx.set_node_attributes(self.G, closeness_centrality, 'closeness')

        # Compute betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(self.G, weight = 'weight', normalized = True)
        nx.set_node_attributes(self.G, betweenness_centrality, 'betweenness')

        return self.G

def save(G, filename = 'roblox'):
    nodes_data = {node: G.nodes[node] for node in G.nodes()}
    df_nodes = pd.DataFrame.from_dict(nodes_data, orient='index')
    df_nodes.reset_index(inplace=True)
    df_nodes.rename(columns={'index': 'Node'}, inplace=True)

    nx.write_graphml(G, f'./../graphs/{filename}_graph.graphml')
    df_nodes.to_csv(f'./../keywords/{filename}_nodes.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action = 'store_true', default = True)
    args = parser.parse_args()

    if args.verbose:
        tqdm.pandas(disable = False)
    else:
        tqdm.pandas(disable = True)

    preprocessor = Preprocessor()
    roblox_df, zepeto_df = preprocessor.run(domain = 'all')

    roblox_analyzer = GraphAnalyzer(roblox_df)
    zepeto_analyzer = GraphAnalyzer(zepeto_df)

    R = roblox_analyzer.run()
    Z = zepeto_analyzer.run()

    save(R, filename = 'roblox')
    save(Z, filename = 'zepeto')

if __name__ == '__main__':
    main()