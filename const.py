import tempfile
import os
import sys

from dep import model_config

spmrl = os.path.expanduser('~/ml_probing/spmrl_2014/')

def encode(language, task, output_dir):
        """
        Encode constituency trees into labels in a format readable by MaChAmp
                inputs: dataset: SPMLR dataset
                        encoding: sequence labeling encoding
                outputs: JSON dict of samples
        """
        # find sets within the dataset
        dataset = spmrl + language.upper() + '_SPMRL/'
        train = dataset + 'gold/ptb/train/train.{}.gold.ptb'.format(language.capitalize())
        dev = dataset + 'gold/ptb/dev/dev.{}.gold.ptb'.format(language.capitalize())
        test = dataset + 'gold/ptb/test/test.{}.gold.ptb'.format(language.capitalize())

        t2l_output = output_dir
        if task == "single":
                encoding_script = 'python tree2labels/dataset.py --train "{}" --dev "{}" --test "{}" --output "{}" --treebank "{}" \
                --encode_unaries --os --abs_top 3 --abs_neg_gap 2'.format(train, dev, test, t2l_output, language)
        
        elif task == "multi":
                encoding_script = 'python tree2labels/dataset.py --train "{}" --dev "{}" --test "{}" --output "{}" --treebank "{}" \
                        --encode_unaries --os --abs_top 3 --abs_neg_gap 2 --split_tags'.format(train, dev, test, t2l_output, language)
        
        os.system(encoding_script)

        # Transform the sequence labeling file into a format readable by MaChAmp if needed
        train_output = t2l_output + '/{}-train.seq_lu'.format(language)
        dev_output = t2l_output + '/{}-dev.seq_lu'.format(language)
        test_output = t2l_output + '/{}-test.seq_lu'.format(language)

        encoded_files = [train_output, dev_output, test_output]
        return encoded_files
    
def decode(file): # TODO
        """
        Encode  labels into dependency trees 
        inputs: file: sequence labeling encoding
                encoding: conllu file
        outputs: conllu file
        """

def dataset_config(language, task, output_dir):
        """
        Create a dataset config file for the dataset
        inputs: treebank: Treebank name
                encoding: sequence labeling encoding
        outputs: dataset config file
        """
        config_file = output_dir + '/dataset_config.json'
        treebank = language.upper() + '_SPMRL/'

        if task == 'single':
                config_json = '''{{
                "SPLMR": {{
                        "train_data_path": "{}",
                        "validation_data_path": "{}",
                        "word_idx": 0,
                        "tasks": 
                        {{
                        "deptag":  {{
                                "task_type": "seq",
                                "column_idx": 2
                                }}
                        }}
                }}
                }}'''.format(
                        encode(language, task, output_dir)[0],
                        encode(language, task, output_dir)[1]
                        )
        elif task == 'multi':
        # TODO: There is no need of multi task setup afaik
                pass

        with open(config_file, 'w') as f:
                f.write(config_json)

        return config_file

def model_config(lm, output_dir):
        """
        Create a model config file for the model
        inputs: lm_model: language model name
        outputs: model config file
        """
        dimension = '768'
        lm = '"{}"'.format(lm)      
        config_template = 'model_config.json'
        config_file = output_dir + '/' + 'model_config.json'
        with open(config_template, 'r') as f:
                config = f.read().replace('transformer_model', lm).replace('transformer_dim', dimension).replace('max_len,', '128,')

        with open(config_file, 'w') as f:
                f.write(config)

        return config_file

def train(language, lm, task, device=0):
    """
    Train a constituency parsing probe using sequence labeling tags
    input: language: language name
              lm: language model
              task: single task or multi task
              device: device to use
    """
    if language == 'english':
        dataset = 'ptb'
    else:
        dataset = 'SPMRL'
        
    model_dir = 'data/const/' + task + '/' + lm
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_dir = os.path.join(model_dir, language)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    name = name = '{}/{}/{}/{}/'.format('const', lm, language, task)
    dataset_config_file = dataset_config(language, task, dataset, model_dir)
    parameters_config_file = model_config(lm, model_dir)
    train_str = 'python3 "machamp/train.py" --dataset_config "{}" --parameters_config "{}" --device "{}" --name "{}" '
    train_str = train_str.format(dataset_config_file, parameters_config_file, device, name)
    os.system(train_str)

def evaluate(language, lm, task, dataset='SPMRL', device=0):
        """
        Evaluate a constituency parsing probe using sequence labeling tags
        input: language: language name
                lm: language model
                device: device to use
        """
        name = '{}/{}/{}/{}/'.format('const', lm, language, task)
        model_dir = '/media/alberto/Seagate Portable Drive/ml_probing/logs/' + name
        model = os.path.join(model_dir + 'model.tar.gz')
        data_dir = 'data/const/' + task + '/' + lm + '/' + language

        # Get test label file
        test_file = data_dir + '/{}-test.seq_lu'.format(language)
        output_seq = model_dir + 'SPLMR.test.out'

        # Predict test file
        predict_str = 'python3 "machamp/predict.py" "{}" "{}" "{}" --dataset SPLMR --device "{}"'
        predict_str = predict_str.format(model, test_file, output_seq, device)
        os.system(predict_str)

        # Get gold test tree file
        treebank = spmrl + language.upper() + '_SPMRL/'
        gold_test_tree = treebank + 'gold/ptb/test/test.{}.gold.ptb'.format(language.capitalize())
        # Convert labels to trees
        #output_tree = model_dir + '/SPLMR.out.tree'
        # Evaluate
        evaluate_str = 'python3 "tree2labels/evaluate.py" --input "{}" --gold "{}" --evalb "{}"'
        if dataset == 'SPMRL':
                evaluate_str = evaluate_str.format(
                        output_seq,
                        gold_test_tree,
                        'tree2labels/EVAL_SPRML/evalb_spmrl2013.final/evalb_spmrl'
                        )
        elif dataset == 'Penn':
                pass #TODO: adjust to Penn Treebank

        os.system(evaluate_str)
        

if __name__ == '__main__':
        evaluate('french', 'bert-base-multilingual-cased', 'single', 'SPMRL', 0)