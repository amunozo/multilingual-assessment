from re import I
import tempfile
import os
import sys
import numpy as np
import conll18_ud_eval as ud_eval
import json
import util
import canine
import subword_models
import shutil

from transformers import AutoModelForTokenClassification

ud = '/home/alberto/Universal Dependencies 2.9/ud-treebanks-v2.9/'
       
def encode(treebank, task, encoding, output_dir):
        """
        Encode dependency trees into labels in a format readable by MaChAmp
        inputs: file: conllu file
                encoding: sequence labeling encoding
        outputs: JSON dict of samples
        """
        # find sets within the dataset
        treebank_dir = ud + treebank + '/'
        encoding_script = 'python3 dep2label/encode_dep2labels.py --input "{}" --output "{}" --encoding "{}"'
        for file in os.listdir(treebank_dir):
                if file.endswith('train.conllu'):
                        conllu_file = treebank_dir + file
                        seq_file = output_dir + 'train'
                        os.system(encoding_script.format(conllu_file, seq_file, encoding))
                elif file.endswith('dev.conllu'):
                        conllu_file = treebank_dir + file
                        seq_file = output_dir + 'dev'
                        os.system(encoding_script.format(conllu_file, seq_file, encoding))
                elif file.endswith('test.conllu'):
                        conllu_file = treebank_dir + file
                        seq_file = output_dir  + 'test'
                        os.system(encoding_script.format(conllu_file, seq_file, encoding))


def decode(file, encoding, task, output_dir, org_conllu): # TODO
        """
        Decode labels into dependency trees 
        inputs: file: sequence labeling encoding
                encoding: conllu file
        outputs: conllu file
        """
        with open(file, 'r') as f:
                if task == 'single':
                        text = f.read()
                elif task == 'multi':
                        lines =  []
                        for line in f.readlines():
                                if line != '\n':
                                        line = line.strip().split('\t')
                                        line = '\t'.join(line[:2]) + '\t' + '{}'.join(line[2:])
                                lines.append(line)
                        text = '\n'.join(lines).replace('\n\n\n', '\n\n')

        d2l_input = tempfile.NamedTemporaryFile(delete=False)
        with open(d2l_input.name, 'w') as f:
                f.write(text)
        
        decoding_script = 'python3 dep2label/decode_labels2dep.py --input "{}" --output "{}" --encoding "{}" --conllu_f "{}"'
        os.system(decoding_script.format(d2l_input.name, output_dir, encoding, org_conllu))

        return output_dir

def train(treebank, lm, finetuned, pretrained, encoding, task='single', not_ft_lr=2e-3, ft_lr = 5e-5, epochs=20,):
        """
        Train a constituency parsing probe using sequence labeling tags
        input: language: language name
                lm: language model
                task: single task or multi task
                device: device to use
        """
        data_dir = 'data/' + encoding + '/' + finetuned + '/' \
                + pretrained + '/' + treebank + '/' + task + '/'


        if not os.path.exists(data_dir):
                os.makedirs(data_dir)

        encode(treebank, task, encoding, data_dir)
        # Create dataset
        util.create_dataset(treebank, finetuned, pretrained, encoding)

        if lm == 'bert-base-multilingual-cased' or lm == 'xlm-roberta-base':
                subword_models.train(
                        treebank=treebank,
                        lm=lm,
                        finetuned=finetuned,
                        pretrained=pretrained,
                        encoding=encoding,
                        not_ft_lr=not_ft_lr,
                        ft_lr=ft_lr,
                        epochs=epochs,
                )
    
        elif lm == 'google/canine-c' or lm == 'google/canine-s':
                canine.train(
                        treebank=treebank,
                        lm=lm,
                        finetuned=finetuned,
                        pretrained=pretrained,
                        encoding=encoding,
                        not_ft_lr=not_ft_lr,
                        ft_lr=ft_lr,
                        epochs=epochs,
                )

def predict(treebank, lm, finetuned, pretrained, encoding, task='single', device=0):
        """
        Predict output test file
        """
        if lm in ('xlm-roberta-base', 'bert-base-multilingual-cased'):
                subword_models.predict(
                        treebank=treebank,
                        lm=lm,
                        finetuned=finetuned,
                        pretrained=pretrained,
                        encoding=encoding,
                        task=task,
                )
        elif lm in ('google/canine-c', 'google/canine-s'):
                canine.predict(
                        treebank=treebank,
                        lm=lm,
                        finetuned=finetuned,
                        pretrained=pretrained,
                        encoding=encoding,
                        task=task,
                )
                
def evaluate(treebank, lm, finetuned, pretrained,
        encoding, task='single'):
        """
        Evaluate a dependency parsing probe using sequence labeling tags
        """
        model_dir = util.ehd_dir + 'models/' + encoding + '/' + finetuned + '/' \
         + pretrained + '/' + lm + '/' + treebank + '/' + task + '/'

        # Find gold conllu file
        ud = '/home/alberto/Universal Dependencies 2.9/ud-treebanks-v2.9/'
        treebank_dir = ud + treebank + '/'
        for file in os.listdir(treebank_dir):
                if file.endswith('ud-test.conllu'):
                        gold_conllu = treebank_dir + file
        # Convert output test file to conllu
        output_dir = model_dir + 'output/'

        output_test = output_dir + 'test.seq'
        output_conllu = output_dir + 'test.conllu'
        decode(output_test, encoding, task, output_conllu, gold_conllu)

        # Evaluate metrics
        dp_metrics = ud_eval.evaluate(
                ud_eval.load_conllu_file(gold_conllu),
                ud_eval.load_conllu_file(output_conllu)
        )

        UAS = dp_metrics['UAS'].f1
        LAS = dp_metrics['LAS'].f1
        CLAS = dp_metrics['CLAS'].f1
        MLAS = dp_metrics['MLAS'].f1
        BLEX = dp_metrics['BLEX'].f1
        
        # Annotate results
        csv_file = 'dep_scores.csv'
        if not os.path.exists(csv_file):
                with open(csv_file, 'w') as f:
                        f.write('Treebank' + ',')
                        f.write('Encoding' + ',')
                        f.write('Language Model' + ',')
                        f.write('Finetuned' + ',')
                        f.write('Pretrained' + ',')
                        f.write('UAS' + ',')
                        f.write('LAS' + ',')
                        f.write('CLAS' + ',')
                        f.write('MLAS' + ',')
                        f.write('BLEX' + ',')
                        f.write('\n')
        
        with open(csv_file, 'a') as f:
                f.write(treebank + ',')
                f.write(encoding + ',')
                f.write(lm + ',')
                f.write(finetuned + ',')
                f.write(pretrained + ',')
                f.write(str(round(UAS*100,2)) + ',')
                f.write(str(round(LAS*100,2)) + ',')
                f.write(str(round(CLAS*100,2)) + ',')
                f.write(str(round(MLAS*100,2)) + ',')
                f.write(str(round(BLEX*100,2)) + ',')
                f.write('\n')
        
        return(UAS, LAS)
                        

def evaluate_displacement(
        treebank,
        encoding,
        finetuned,
        pretrained, 
        lms = ['bert-base-multilingual-cased', 'xlm-roberta-base', 'google/canine-c', 'google/canine-s'],
        ):
        """
        Evaluate and plot the dependency displacements of a treebank
        """
        # Locate gold conllu file
        ud = '/home/alberto/Universal Dependencies 2.9/ud-treebanks-v2.9/'
        treebank_dir = ud + treebank + '/'
        for file in os.listdir(treebank_dir):
                if file.endswith('ud-test.conllu'):
                        gold_conllu = treebank_dir + file

        # Locate test.conllu files
        test_conllus = []
        for lm in lms:
                model_dir = util.ehd_dir + 'models/' + encoding + '/' + finetuned + '/' \
                 + pretrained + '/' + lm + '/' + treebank + '/single/'
                output_dir = model_dir + 'output/'
                output_conllu = output_dir + 'test.conllu'
                test_conllus.append(output_conllu)
        
        # Create a temporary directory to store the test.conllu files
        temp_dir = 'temp/'
        if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
        for test_conllu, lm in zip(test_conllus, lms):
                if lm.startswith('google'):
                        lm = lm.split('/')[-1]
                shutil.copy(test_conllu, temp_dir + lm)
        shutil.copy(gold_conllu, temp_dir + 'gold')
        # Define output directory
        output_dir = 'plots/displacements/' +  encoding + '/' + finetuned + '/' + pretrained + '/' + treebank + '/'
        if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        output = output_dir + 'displacements.png'
        # Evaluate displacements
        disp_script = 'python evaluate_dependencies.py --gold "{}" --predicted "{}" --output "{}"'.format(gold_conllu, temp_dir, output)
        os.system(disp_script)

        # Remove temporary directory
        #shutil.rmtree(temp_dir)

def evaluate_avg_displacement(
        treebanks,
        encoding,
        finetuned,
        pretrained, 
        lms = ['bert-base-multilingual-cased', 'xlm-roberta-base', 'google/canine-c', 'google/canine-s'],
        ):
        """
        Evaluate and plot the dependency displacements of a treebank
        """
        # Locate gold conllu file
        ud = '/home/alberto/Universal Dependencies 2.9/ud-treebanks-v2.9/'
        treebank_dir = ud + treebank + '/'
        for file in os.listdir(treebank_dir):
                if file.endswith('ud-test.conllu'):
                        gold_conllu = treebank_dir + file

        # Locate test.conllu files
        test_conllus = []
        for lm in lms:
                model_dir = util.ehd_dir + 'models/' + encoding + '/' + finetuned + '/' \
                 + pretrained + '/' + lm + '/' + treebank + '/single/'
                output_dir = model_dir + 'output/'
                output_conllu = output_dir + 'test.conllu'
                test_conllus.append(output_conllu)
        
        # Create a temporary directory to store the test.conllu files
        temp_dir = 'temp/'
        if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
        for test_conllu, lm in zip(test_conllus, lms):
                if lm.startswith('google'):
                        lm = lm.split('/')[-1]
                shutil.copy(test_conllu, temp_dir + lm)
        
        # Define output directory
        output_dir = 'plots/displacements/' +  encoding + '/' + finetuned + '/' + pretrained + '/' + treebank + '/'
        if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        output = output_dir + 'displacements.png'
        # Evaluate displacements
        disp_script = 'python evaluate_dependencies.py --gold "{}" --predicted "{}" --output "{}"'.format(gold_conllu, temp_dir, output)
        os.system(disp_script)

        # Remove temporary directory
        shutil.rmtree(temp_dir)
if __name__ == '__main__':
        treebanks = [
    'UD_Ancient_Greek-Perseus', 'UD_Skolt_Sami-Giellagas', 'UD_Welsh-CCG',
    'UD_Bulgarian-BTB', 'UD_Guajajara-TuDeT', 'UD_Armenian-ArmTDP',
    'UD_Turkish-BOUN', 'UD_Ligurian-GLT', 'UD_Vietnamese-VTB',
    'UD_Basque-BDT', 'UD_Bhojpuri-BHTB', 'UD_Kiche-IU', 'UD_Chinese-GSDSimp',
        ]
        for treebank in treebanks:
                evaluate_displacement(treebank, '2-planar-brackets-greedy', 'not_finetuned', 'pretrained')