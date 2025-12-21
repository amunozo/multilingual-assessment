import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dep import train, predict, evaluate

mbert_dic = {
    'UD_Ancient_Greek-Perseus': False,
    'UD_Skolt_Sami-Giellagas': False,
    'UD_Welsh-CCG': True,
    'UD_Bulgarian-BTB': True,
    'UD_Guajajara-TuDeT': False,
    'UD_Armenian-ArmTDP': True,
    'UD_Turkish-BOUN': True,
    'UD_Ligurian-GLT': False,
    'UD_Vietnamese-VTB': True,
    'UD_Basque-BDT': True,
    'UD_Bhojpuri-BHTB': False,
    'UD_Kiche-IU': False,
}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_dep.py <device>")
        sys.exit(1)
        
    task = 'single'
    device = sys.argv[1]
    encodings = ['2-planar-brackets-greedy', 'relative', 'absolute', 'rel-pos']
    treebanks = []
    
    for treebank in treebanks:
        for pretrained in ['pretrained', 'not_pretrained']:
            if pretrained == 'not_pretrained':
                lms = ['random_models/bert-base-multilingual-cased', 'random_models/xlm-roberta-base']
            elif pretrained == 'pretrained':
                lms = ['bert-base-multilingual-cased', 'xlm-roberta-base']

            for lm in lms:
                for encoding in encodings:
                    for finetuned in ['finetuned', 'not_finetuned']:
                        try:
                            train(treebank, lm, finetuned, pretrained, encoding)
                            predict(treebank, lm, finetuned, pretrained, encoding, device=device)
                            evaluate(treebank, lm, finetuned, pretrained, encoding)
                        except Exception as e:
                            print(f"Error with {treebank} {lm} {encoding} {finetuned}: {e}")
