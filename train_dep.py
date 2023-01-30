import sys
from dep import train, evaluate

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
            'UD_Kiche-IU': False
        } # es igual que el de xlm-roberta-base

if __name__ == '__main__':
        #encoding = sys.argv[1]
        #lm = sys.argv[2] #bert-base-multilingual-cased, 'xlm-roberta-base'
        task = sys.argv[1]
        mode = sys.argv[2]
        device = sys.argv[3]
        encodings = ['2-planar-brackets-greedy', 'relative', 'absolute', 'rel-pos']
        treebanks = [
    'UD_Guajajara-TuDeT', 'UD_Skolt_Sami-Giellagas', 'UD_Welsh-CCG',
    'UD_Bulgarian-BTB','UD_Ancient_Greek-Perseus' , 'UD_Armenian-ArmTDP',
    'UD_Turkish-BOUN', 'UD_Ligurian-GLT', 'UD_Vietnamese-VTB',
    'UD_Basque-BDT', 'UD_Bhojpuri-BHTB', 'UD_Kiche-IU'
        ]
        if mode == 'random':
            lms = ['random_models/bert-base-multilingual-cased', 'random_models/xlm-roberta-base']
        elif mode == 'pretrained':
            lms = ['bert-base-multilingual-cased', 'xlm-roberta-base']

        for treebank in treebanks:
            for lm in lms:
                for encoding in encodings:
                    train(treebank, encoding, lm, task, device) 
                    evaluate(treebank, encoding, lm, task, mbert_dic[treebank], device=device)