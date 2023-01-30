import const
import dep
import torch

const_language_models = [
    #'bert-base-multilingual-cased',
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
ud_treebanks = [
    #'UD_Ligurian-GLT',
    #'UD_Bhojpuri-BHTB',
    #'UD_Kiche-IU',
    #'UD_Guajajara-TuDeT',
    #'UD_Skolt_Sami-Giellagas',
    #'UD_Armenian-ArmTDP',
    #'UD_Welsh-CCG',
    #'UD_Vietnamese-VTB',
    #'UD_Basque-BDT',
    #'UD_Bulgarian-BTB',
    #'UD_Turkish-BOUN', 
    #'UD_Ancient_Greek-Perseus',
    #'UD_Chinese-GSDSimp', 
    ]

ud_treebanks_test = [
    'UD_Classical_Chinese-Kyoto',
    'UD_Naija-NSC',
    'UD_Maltese-MUDT',
    'UD_Gothic-PROIEL',
    'UD_Wolof-WTB',
    'UD_Old_East_Slavic-TOROT',
]

const_languages = [
    #'french', 
    #'basque',
    #'hebrew',  
    #'hungarian', 
    #'korean', 
    #'polish', 
    #'swedish',
    #'english',
    #'chinese',
    #'german'
]

torch.cuda.empty_cache()
with open('log_dep.txt', 'w') as f:
    f.write('Start training')

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
                    with open('log_dep.txt', 'a') as f:
                       f.write('Done with {} {} {} {} {}\n'.format(treebank, lm, encoding, finetuned, pretrained))
                except:
                    print('Error with {} {} {} {} {}'.format(treebank, lm, encoding, finetuned, pretrained))

with open('log_const.txt', 'w') as f:
    f.write('Start training\n')

# Constituency
# Swedish and Hebrew: train5k instead of train
for language in const_languages:
   for lm in const_language_models:
        for finetuned, pretrained in const_setup:
            torch.cuda.empty_cache()
            #const.train(language, lm, finetuned, pretrained, epochs=20, not_ft_lr=5e-4)
            const.predict(language, lm, finetuned, pretrained) #, epochs=20)
            const.evaluate(language, lm, finetuned, pretrained) #, epochs=20)
            torch.cuda.empty_cache()
            with open('log_const.txt', 'a') as f:
                f.write('Done with {} {} {} {}\n'.format(language, lm, finetuned, pretrained))
