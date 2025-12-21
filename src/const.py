import tempfile
import os
import sys
import subword_models
import util
import canine
import torch
import shutil

spmrl = os.path.expanduser('~/ml_probing/spmrl_2014/')

def encode(language, task, output_dir):
        """
        Encode constituency trees into labels in a format readable by MaChAmp
                inputs: dataset: SPMLR dataset
                        encoding: sequence labeling encoding
                outputs: JSON dict of samples
        """
        # find sets within the dataset
        if language == 'english':
                dataset = 'PTB/'
                train = dataset + 'train.trees'
                dev = dataset + 'dev.trees'
                test = dataset + 'test.trees'

        elif language == 'chinese':
                dataset = 'CTB/'
                train = dataset + 'train_ch.trees'
                dev = dataset + 'dev_ch.trees'
                test = dataset + 'test_ch.trees'
        else:
                dataset = spmrl + language.upper() + '_SPMRL/'
                train = dataset + 'gold/ptb/train/train.{}.gold.ptb'.format(language.capitalize())
                dev = dataset + 'gold/ptb/dev/dev.{}.gold.ptb'.format(language.capitalize())
                test = dataset + 'gold/ptb/test/test.{}.gold.ptb'.format(language.capitalize())

        
        

        t2l_output = output_dir

        train_output = t2l_output + '/{}-train.seq_lu'.format(language)
        dev_output = t2l_output + '/{}-dev.seq_lu'.format(language)
        test_output = t2l_output + '/{}-test.seq_lu'.format(language)       

        encoding_script = 'python tree2labels/dataset.py --train "{}" --dev "{}" --test "{}" --treebank "{}" --output "{}" --os --encode_unaries'.format(train, dev, test, language, t2l_output)
  
        os.system(encoding_script)
        
        encoded_files = [train_output, dev_output, test_output]
        return encoded_files
    
def decode(file, trees, output): # TODO
        """
        Encode  labels into dependency trees 
        inputs: file: sequence labeling encoding
                encoding: conllu file
        outputs: conllu file
        """
        decoding_script = 'python tree2labels/decode.py --input "{}" --gold "{}" --output "{}"'.format(file, trees, output)
        os.system(decoding_script)

def train(language, lm, finetuned, pretrained, 
        encoding='const', task='single', not_ft_lr=2e-3, ft_lr = 5e-5, epochs=10,):
    """
    Train a constituency parsing probe using sequence labeling tags
    input: language: language name
              lm: language model
              task: single task or multi task
              device: device to use
    """
    data_dir = 'data/' + encoding + '/' + finetuned + '/' \
         + pretrained + '/' + language + '/' + task + '/'


    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    encode(language, task, data_dir)

    # Create dataset
    util.create_dataset(language, finetuned, pretrained, encoding)

    if lm == 'bert-base-multilingual-cased' or lm == 'xlm-roberta-base':
        subword_models.train(
                treebank=language,
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
                treebank=language,
                lm=lm,
                finetuned=finetuned,
                pretrained=pretrained,
                encoding=encoding,
                not_ft_lr=not_ft_lr,
                ft_lr=ft_lr,
                epochs=epochs,
        )

def predict(treebank, lm, finetuned, pretrained, encoding='const', task='single'):
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

def evaluate(language, lm, finetuned, pretrained, encoding='const', task='single', device=0):
        """
        Evaluate a constituency parsing probe using sequence labeling tags
        input: language: language name
                lm: language model
                device: device to use
        """
        model_dir = util.ehd_dir + 'models/' + encoding + '/' + finetuned + '/' \
         + pretrained + '/' + lm + '/' + language + '/' + task + '/'
        output_dir = model_dir + 'output/'
        output_test = output_dir + 'test.seq'
        evaluate_str = 'python3 "tree2labels/evaluate.py" --input "{}" --gold "{}" --evalb "{}"'
                
        # Get gold test tree file

        if language == 'english' or language == 'chinese':
                # Get test label file
                if language == 'english':
                        treebank = 'PTB/'
                        gold_test_tree = treebank + 'test.trees'
                elif language == 'chinese':
                        treebank = 'CTB/'
                        gold_test_tree = treebank + 'test_ch.trees'
                # Evaluate
                evaluate_str = evaluate_str.format(
                        output_test,
                        gold_test_tree,
                        'tree2labels/EVALB/evalb'
                        )
        else:
                # Get test label file
                treebank = spmrl + language.upper() + '_SPMRL/'
                gold_test_tree = treebank + 'gold/ptb/test/test.{}.gold.ptb'.format(language.capitalize())
                # Evaluate        
                evaluate_str = evaluate_str.format(
                        output_test,
                        gold_test_tree,
                        'tree2labels/EVAL_SPRML/evalb_spmrl2013.final/evalb_spmrl'
                        )
        
        
        output = os.popen(evaluate_str).readlines()
        results = output[-26:-16]
        results_dic = {}
        for line in results:
                line = line.replace(' ', '').replace('\n', '').split('=')
                results_dic[line[0]] = line[1]
        
        if not os.path.exists('const_scores.csv'):
                with open('const_scores.csv', 'w') as f:
                        f.write('language,lm,finetuned,pretrained,encoding,task,')
                        for key in results_dic.keys():
                                f.write(key + ',')
                        f.write('\n')

        with open('const_scores.csv', 'a') as f:
                f.write(language + ',' + lm + ',' + finetuned + ',' + pretrained + ',' + encoding + ',' + task + ',')
                for key in results_dic.keys():
                        f.write(results_dic[key] + ',')
                f.write('\n')

def evaluate_spans(
        treebank,
        encoding,
        finetuned,
        pretrained, 
        lms = ['bert-base-multilingual-cased', 'xlm-roberta-base', 'google/canine-c', 'google/canine-s'],
        ):
        """
        Evaluate and plot the dependency displacements of a treebank
        """
        # Locate gold file
        if language == 'english':
                dataset = 'PTB/'
                gold = os.path.abspath(dataset + 'test.trees')

        elif language == 'chinese':
                dataset = 'CTB/'
                gold = os.path.abspath(dataset + 'test_ch.trees')
        else:
                dataset = spmrl + language.upper() + '_SPMRL/'
                gold = os.path.abspath(
                       dataset + 'gold/ptb/test/test.{}.gold.ptb'.format(language.capitalize())
                )
        

        # Locate test.conllu files
        test_trees = []
        for lm in lms:
                model_dir = util.ehd_dir + 'models/' + encoding + '/' + finetuned + '/' \
                 + pretrained + '/' + lm + '/' + treebank + '/single/'
                output_dir = model_dir + 'output/'
                output_seq = output_dir + 'test.seq'
                # convert to .trees
                output_tree = output_dir + 'test.trees'
                decode(output_seq, gold, output_tree)

                test_trees.append(output_tree)
        
        # Create a temporary directory to store the test.conllu files
        temp_dir = 'temp/'
        if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)


        for test_conllu, lm in zip(test_trees, lms):
                if lm.startswith('google'):
                        lm = lm.split('/')[-1]
                shutil.copy(test_conllu, temp_dir + lm)
        
        # Define output directory
        output_dir = 'plots/spans/' +  encoding + '/' + finetuned + '/' + pretrained + '/' + treebank + '_'
        if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        output_dir_nt = output_dir.replace('spans', 'non_terminal')
        if not os.path.exists(output_dir_nt):
                os.makedirs(output_dir_nt)
        output = output_dir + 'spans.png'
        # Evaluate displacements
        disp_script = 'python evaluate_spans.py --gold "{}" --predicted "{}" --output "{}"'.format(gold, temp_dir, output)
        os.system(disp_script)

        # Remove temporary directory
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
        languages = [
                'korean',
                'english',
                #'chinese',
                'basque', 
                #'french',
                #'german',
                #'hebrew',
                'hungarian',
                #'polish',
                #'swedish',
        ]
        for language in languages:
                evaluate_spans(language, 'const', 'not_finetuned', 'pretrained')
