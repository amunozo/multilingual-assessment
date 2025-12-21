import os
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import const
from src import dep

const_language_models = [
    'bert-base-multilingual-cased',
    'xlm-roberta-base',
]

dep_language_models = [
    'google/canine-s',
]

dep_encodings = [
    '2-planar-brackets-greedy', 
    'rel-pos',
    'arc-hybrid',
]
const_encodings = ['const']
setup = [
    ('finetuned', 'pretrained'),
    ('not_finetuned', 'pretrained'),
    ('not_finetuned', 'not_pretrained')
]
ud_treebanks = [
    'UD_Ancient_Greek-Perseus', 
]

const_languages = [
    'english',
    'german'
]

log_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(log_dir, exist_ok=True)

log_dep_path = os.path.join(log_dir, 'log_dep.txt')
with open(log_dep_path, 'w') as f:
    f.write('Start evaluation\n')

# Dependencies
for treebank in ud_treebanks:
    for lm in dep_language_models:
        for encoding in dep_encodings:
            for finetuned, pretrained in setup:
                try:
                    dep.predict(treebank, lm, finetuned, pretrained, encoding)
                    dep.evaluate(treebank, lm, finetuned, pretrained, encoding)
                    with open(log_dep_path, 'a') as f:
                        f.write(f'Done with {treebank} {lm} {encoding} {finetuned} {pretrained}\n')
                except Exception as e:
                    print(f"Error with {treebank} {lm} {encoding} {finetuned} {pretrained}: {e}")