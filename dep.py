from re import I
import tempfile
import os
import sys
import numpy as np
import conll18_ud_eval as ud_eval
import json

ud = '/home/alberto/Universal Dependencies 2.9/ud-treebanks-v2.9/'
       
def encode(file, encoding, task, output_dir):
        """
        Encode dependency trees into labels in a format readable by MaChAmp
        inputs: file: conllu file
                encoding: sequence labeling encoding
        outputs: JSON dict of samples
        """
        d2l_output = tempfile.NamedTemporaryFile(delete=False)
        encoding_script = 'python3 dep2label/encode_dep2labels.py --input "{}" --output "{}" --encoding "{}"'.format(file, d2l_output.name, encoding)
        os.system(encoding_script)

        # Transform the sequence labeling file into a format readable by MaChAmp if needed
        with open(d2l_output.name, 'r') as f:
                if task == 'multi':
                        text = f.read().replace('{}', '\t')
                        #text = []
                        #for line in f.readlines():
                        #        if line != '\n':
                        #                line = line.replace('{}', '\t')
                        #                line = line.strip().split('\t')
                        #                line = '\t'.join(line[:2]) + '\t' + '{}'.join(line[2:4]) + '\t' + line[4]
                        #        text.append(line)
                        #text = '\n'.join(text).replace('\n\n\n', '\n\n')
                elif task == 'single':
                        text = f.read()
        
        if file.endswith('train.conllu'):
                seq_machamp = output_dir + 'train'
        elif file.endswith('dev.conllu'):
                seq_machamp = output_dir + 'dev'
        elif file.endswith('test.conllu'):
                seq_machamp = output_dir  + 'test'

        with open(seq_machamp, 'w') as f:
                f.write(text)

        return seq_machamp

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

def dataset_config(treebank, encoding, task, output_dir):
        """
        Create a dataset config file for the dataset
        inputs: treebank: Treebank name
                encoding: sequence labeling encoding
        outputs: dataset config file
        """

        config_file = output_dir + '/dataset_config.json'
        treebank_dir = os.path.join(ud, treebank)
        for f in os.listdir(treebank_dir):
                if f.endswith('train.conllu'):
                        train_conllu = os.path.join(treebank_dir, f)
                elif f.endswith('dev.conllu'):
                        dev_conllu = os.path.join(treebank_dir, f)
        if task == 'single':
                config_json = '''{{
                "UD": {{
                        "train_data_path": "{}",
                        "validation_data_path": "{}",
                        "word_idx": 0,
                        "tasks":
                        {{
                        "postag":  {{
                                "task_type": "seq",
                                "column_idx": 1
                        }},
                        "deptag":  {{
                                "task_type": "seq",
                                "column_idx": 2
                                }}
                        }}
                }}
                }}'''.format(
                        encode(train_conllu, encoding, task, output_dir),
                        encode(dev_conllu, encoding, task, output_dir)
                        )
        elif task == 'multi':
                if encoding == '2-planar-brackets-greedy':
                        config_json = '''{{
                "UD": {{
                        "train_data_path": "{}",
                        "validation_data_path": "{}",
                        "word_idx": 0,
                        "tasks": 
                        {{
                        "postag":  {{
                                "task_type": "seq",
                                "column_idx": 1
                        }},
                        "1-plane": {{
                                "task_type": "seq",
                                "column_idx": 2
                        }},
                        "2-plane": {{
                                "task_type": "seq",
                                "column_idx": 3
                        }},
                        "deprel": {{
                                "task_type": "seq",
                                "column_idx": 4
                        }}
                        }}
                }}
                }}'''.format(
                        encode(train_conllu, encoding, task, output_dir),
                        encode(dev_conllu, encoding, task, output_dir)
                        )

                else:
                        config_json = '''{{
                "UD": {{
                        "train_data_path": "{}",
                        "validation_data_path": "{}",
                        "word_idx": 0,
                        "tasks": 
                        {{
                        "postag":  {{
                                "task_type": "seq",
                                "column_idx": 1
                        }},
                        "arc": {{
                                "task_type": "seq",
                                "column_idx": 2
                        }},
                        "deprel": {{
                                "task_type": "seq",
                                "column_idx": 3
                        }}
                        }}
                }}
                }}'''.format(
                        encode(train_conllu, encoding, task, output_dir),
                        encode(dev_conllu, encoding, task, output_dir)
                        )

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
        
        if lm.startswith('random_models'):
                random = True
                config_template = 'random_config.json'
                random_model = lm
                lm = lm.split('/')[-1]
        else:
                random = False
                config_template = 'model_config.json'
        
        lm = '"{}"'.format(lm)
        config_file = output_dir + '/' + 'model_config.json'
        with open(config_template, 'r') as f:
                config = f.read().replace('transformer_model', lm).replace('transformer_dim', dimension).replace('max_len,', '128,')
                if random:
                        config = config.replace('random_model', '"{}"'.format(random_model))

        with open(config_file, 'w') as f:
                f.write(config)

        return config_file

def train(treebank, encoding, lm, task, device=0):
       """
       Train a dependency parsing probe using sequence labeling tags
       input: treebank:Treebank name
              encoding: sequence labeling encoding name
              lm: huggingface language model name
              task: single task or multi task
              device: device to use
       """
       model_dir = 'data' + '/' + encoding + '/' + task + '/' + lm  + '/' 
       if not os.path.exists(model_dir):
              os.makedirs(model_dir)

       model_dir = model_dir + '/' + treebank + '/'
       if not os.path.exists(model_dir):
              os.makedirs(model_dir)

       name = '{}/{}/{}/{}/'.format(encoding, lm, treebank, task)
       dataset_config_file = dataset_config(treebank, encoding, task, model_dir)
       parameters_config_file = model_config(lm, model_dir)
       # xlm-roberta dimension is 1024; bert is 768

       train_str = 'python3 "machamp/train.py" --dataset_config "{}" --parameters_config "{}" --device "{}" --name "{}" '
       train_str = train_str.format(dataset_config_file, parameters_config_file, device, name)
       os.system(train_str)

def evaluate(treebank, encoding, lm, task, in_training_data, device=0):
        """
        Evaluate a dependency parsing probe using the CoNLL evaluation script
        input: treebank:Treebank name
               encoding: sequence labeling encoding name
               lm: huggingface language model name
               task: single task or multi task
               device: device to use
        """
        model_dir = '/media/alberto/Seagate Portable Drive/ml_probing/logs/' + '{}/{}/{}/{}/'.format(encoding, lm, treebank, task)
        model = os.path.join(model_dir + 'model.tar.gz')
        
        # Get test file
        treebank_dir = os.path.join(ud, treebank)
        for f in os.listdir(treebank_dir):
                if f.endswith('test.conllu'):
                        gold_conllu = os.path.join(treebank_dir, f)
        
        test_file = encode(gold_conllu, encoding, task, model_dir)# + 'test')
        output_seq = model_dir + '/UD.test.out'
        
        # Predict test file
        predict_str = 'python3 "machamp/predict.py" "{}" "{}" "{}" --dataset UD --device "{}"'
        predict_str = predict_str.format(model, test_file, output_seq, device)
        os.system(predict_str)

        # Convert to CoNLLu
        output_conllu = model_dir + '/output.conllu'
        decode(output_seq, encoding, task, output_conllu, gold_conllu)

        if lm.startswith('random_models'):
                lm = lm.split('/')[-1]
                random = True
        else:
                random = False

        # Evaluate metrics
        dp_metrics = ud_eval.evaluate(
                ud_eval.load_conllu_file(gold_conllu),
                ud_eval.load_conllu_file(output_conllu)
        )

        with open(model_dir + '/UD.test.out.eval', 'r') as f:
                tag_metrics = json.loads(f.read().replace("'", "\""))
        if task == 'multi':
                if encoding == '2-planar-brackets-greedy':
                        first = tag_metrics['.run/1-plane/acc']
                        second = tag_metrics['.run/2-plane/acc']
                        arc = np.nan
                else:   
                        first = np.nan
                        second = np.nan
                        arc = tag_metrics['.run/arc/acc']
                deprel = tag_metrics['.run/deprel/acc']
                deptag = np.nan

        elif task == 'single':
                first = np.nan
                second = np.nan
                arc = np.nan
                deprel = np.nan
                deptag = tag_metrics['.run/deptag/acc']
        
        postag = tag_metrics['.run/postag/acc']
        UAS = dp_metrics['UAS'].f1
        LAS = dp_metrics['LAS'].f1
        CLAS = dp_metrics['CLAS'].f1
        MLAS = dp_metrics['MLAS'].f1
        BLEX = dp_metrics['BLEX'].f1
        
        # Annotate results
        csv_file = 'scores.csv'
        if not os.path.exists(csv_file):
                with open(csv_file, 'w') as f:
                        f.write('Treebank' + ',')
                        f.write('Encoding' + ',')
                        f.write('Language Model' + ',')
                        f.write('In training data' + ',')
                        f.write('Random' + ',')
                        f.write('Task' + ',')
                        f.write('UAS' + ',')
                        f.write('LAS' + ',')
                        f.write('CLAS' + ',')
                        f.write('MLAS' + ',')
                        f.write('BLEX' + ',')
                        f.write('POS' + ',')
                        f.write('1-deptag' + ',')
                        f.write('Arc' + ',')
                        f.write('1-plane' + ',')
                        f.write('2-plane' + ',')
                        f.write('Deprel')
                        f.write('\n')
        
        with open(csv_file, 'a') as f:
                f.write(treebank + ',')
                f.write(encoding + ',')
                f.write(lm + ',')
                f.write(str(in_training_data) + ',')
                f.write(str(random) + ',')
                f.write(task + ',')
                f.write(str(round(UAS*100,2)) + ',')
                f.write(str(round(LAS*100,2)) + ',')
                f.write(str(round(CLAS*100,2)) + ',')
                f.write(str(round(MLAS*100,2)) + ',')
                f.write(str(round(BLEX*100,2)) + ',')
                f.write(str(round(postag*100,2)) + ',')
                f.write(str(round(deptag*100,2)) + ',')
                f.write(str(round(arc*100,2)) + ',')
                f.write(str(round(first*100,2)) + ',')
                f.write(str(round(second*100,2)) + ',')
                f.write(str(round(deprel*100,2)))
                f.write('\n')
                        


if __name__ == '__main__':
        treebanks = [
    'UD_Ancient_Greek-Perseus', 
    #'UD_Skolt_Sami-Giellagas', 'UD_Welsh-CCG',
    #'UD_Bulgarian-BTB', 'UD_Guajajara-TuDeT', 'UD_Armenian-ArmTDP',
    #'UD_Turkish-BOUN', 'UD_Ligurian-GLT', 'UD_Vietnamese-VTB',
    #'UD_Basque-BDT', 'UD_Bhojpuri-BHTB', 'UD_Kiche-IU'
        ]

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
        }
        for encoding in ['2-planar-brackets-greedy', 'absolute', 'rel-pos', 'relative']:
                treebank = 'UD_Guajajara-TuDeT'
                train('UD_Guajajara-TuDeT', encoding, 'bert-base-multilingual-cased', 'multi', 0)
                evaluate('UD_Guajajara-TuDeT', encoding, 'bert-base-multilingual-cased', 'multi', mbert_dic[treebank], 0)