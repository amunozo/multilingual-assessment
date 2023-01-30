import const
import dep

const_language_models = [
    #'google/canine-s',
    #'google/canine-c',
    'bert-base-multilingual-cased',
    'xlm-roberta-base',
]

dep_language_models = [
    'google/canine-s',
    #'bert-base-multilingual-cased',
    #'xlm-roberta-base',
    #'google/canine-c',
]

dep_encodings = [
    '2-planar-brackets-greedy', 
    'rel-pos',
    'arc-hybrid',
    #'covington'
]
const_encodings = ['const']
setup = [
    ('finetuned', 'pretrained'),
    ('not_finetuned', 'pretrained'),
    ('not_finetuned', 'not_pretrained')
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
    'UD_Ancient_Greek-Perseus', 
    ]

const_languages = [
    # 'french', 
    # 'basque',
    # 'hebrew',  
    # 'hungarian', 
    #'korean', 
    #'polish', 
    #'swedish',
    'english',
    'german'
]

#with open('log_const.txt', 'w') as f:
#    f.write('Start training')
# Constituency
# Swedish and Hebrew: train5k instead of train
#for language in const_languages:
#    for lm in const_language_models:
#        for finetuned, pretrained in setup:
#            const.train(language, lm, finetuned, pretrained, epochs=20)
#            with open('log_const.txt', 'a') as f:
#                f.write('Done with {} {} {} {}\n'.format(language, lm, finetuned, pretrained))

with open('log_dep.txt', 'w') as f:
    f.write('Start training')
# Dependencies
for treebank in ud_treebanks:
    for lm in dep_language_models:
        for encoding in dep_encodings:
            for finetuned, pretrained in setup:
                    dep.predict(treebank, lm, finetuned, pretrained, encoding)
                    dep.evaluate(treebank, lm, finetuned, pretrained, encoding)
                    with open('log_dep.txt', 'a') as f:
                        f.write('Done with {} {} {} {} {}\n'.format(treebank, lm, encoding, finetuned, pretrained))




                #dep.predict(treebank, lm, finetuned, pretrained,encoding)
                #UAS, LAS =dep.evaluate(treebank, lm, finetuned, pretrained,encoding)
                #print('Done with', treebank, encoding, lm, finetuned, pretrained)
                #print('UAS:', UAS, 'LAS:', LAS)



#const.train(
#    'hebrew',
#    'google/canine-c',
#    'finetuned',
#    'pretrained',
#    epochs=20,
#)