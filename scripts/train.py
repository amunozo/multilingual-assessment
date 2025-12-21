import torch
import os
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import const
from src import dep

const_language_models = [
    'xlm-roberta-base',
    'google/canine-c',
    'google/canine-s',
]

dep_language_models = [
    'google/canine-s',
    'google/canine-c',
    'bert-base-multilingual-cased',
    'xlm-roberta-base',
]

dep_encodings = [
    '2-planar-brackets-greedy', 
    'arc-hybrid',
    'relative'
]
const_encodings = ['const']
const_setup = [
    ('not_finetuned', 'not_pretrained'),
    ('not_finetuned', 'pretrained'),
    ('finetuned', 'pretrained'),
]
dep_setup = [
    ('not_finetuned', 'not_pretrained'),
    ('not_finetuned', 'pretrained'),
    ('finetuned', 'pretrained'),
]

ud_treebanks_test = [
    'UD_Classical_Chinese-Kyoto',
    'UD_Naija-NSC',
    'UD_Maltese-MUDT',
    'UD_Gothic-PROIEL',
    'UD_Wolof-WTB',
    'UD_Old_East_Slavic-TOROT',
]

const_languages = []

torch.cuda.empty_cache()

log_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(log_dir, exist_ok=True)

log_dep_path = os.path.join(log_dir, 'log_dep.txt')
with open(log_dep_path, 'w') as f:
    f.write('Start training\n')

# Dependencies
for treebank in ud_treebanks_test:
    for lm in dep_language_models:
        for encoding in dep_encodings:
            for finetuned, pretrained in dep_setup:
                try:
                    dep.train(treebank, lm, finetuned, pretrained, encoding, epochs=20)
                    dep.predict(treebank, lm, finetuned, pretrained, encoding)
                    dep.evaluate(treebank, lm, finetuned, pretrained, encoding)
                    torch.cuda.empty_cache()
                    with open(log_dep_path, 'a') as f:
                       f.write(f'Done with {treebank} {lm} {encoding} {finetuned} {pretrained}\n')
                except Exception as e:
                    print(f'Error with {treebank} {lm} {encoding} {finetuned} {pretrained}: {e}')

log_const_path = os.path.join(log_dir, 'log_const.txt')
with open(log_const_path, 'w') as f:
    f.write('Start training\n')

# Constituency
for language in const_languages:
   for lm in const_language_models:
        for finetuned, pretrained in const_setup:
            torch.cuda.empty_cache()
            try:
                const.predict(language, lm, finetuned, pretrained)
                const.evaluate(language, lm, finetuned, pretrained)
                torch.cuda.empty_cache()
                with open(log_const_path, 'a') as f:
                    f.write(f'Done with {language} {lm} {finetuned} {pretrained}\n')
            except Exception as e:
                print(f'Error with {language} {lm} {finetuned} {pretrained}: {e}')
