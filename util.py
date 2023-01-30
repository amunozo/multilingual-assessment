from datasets import Sequence, Value, ClassLabel, Features, load_dataset, Features
import os
import conllu
import json
from tempfile import NamedTemporaryFile



ehd_dir = "/media/alberto/Seagate Portable Drive/ml_probing/"


def load_conllu(filename):
    with open(filename) as f:
        data = conllu.parse(f.read())
    return data

def create_json_files(data_files):
    """Convert the TSV data files in a JSON file to create a dataset"""
    train_json = NamedTemporaryFile("w", delete=False)
    eval_json = NamedTemporaryFile("w", delete=False)
    test_json = NamedTemporaryFile("w", delete=False) 

    train_list = []
    eval_list = []
    test_list = []
    
    class_names = return_class_names(data_files)

    for split in data_files:
        idx = 0
        file = data_files[split]
        with open(file, 'r') as f:
            sentences = f.read().split('\n\n')
            for sentence in sentences:
                idx += 1
                sentence_tokens = []
                sentence_pos_tags = []
                sentence_syntax_labels = []

                lines = sentence.split('\n')
                if len(lines) > 1:
                    for line in lines:
                        if line:
                            token, pos_tag, syntax_label = line.split('\t')
                            sentence_tokens.append(token)
                            sentence_pos_tags.append(pos_tag)
                            if split != "test":
                                sentence_syntax_labels.append(syntax_label)
                            else:
                                if syntax_label in class_names["syntax_labels"]:
                                    sentence_syntax_labels.append(syntax_label)
                                else:
                                    sentence_syntax_labels.append("UNK")


                if sentence_tokens != []:
                    if split == "train":
                        train_list.append({"id": idx, "tokens": sentence_tokens, "pos_tags": sentence_pos_tags, "syntax_labels": sentence_syntax_labels})
                    elif split == "validation":
                        eval_list.append({"id": idx, "tokens": sentence_tokens, "pos_tags": sentence_pos_tags, "syntax_labels": sentence_syntax_labels})
                    else:
                        test_list.append({"id": idx, "tokens": sentence_tokens, "pos_tags": sentence_pos_tags, "syntax_labels": sentence_syntax_labels})

    
    train_dic = {"data": train_list}
    eval_dic = {"data": eval_list}
    test_dic = {"data": test_list}

    with open(train_json.name, 'w', encoding='utf8') as f:
        json.dump(train_dic, f, ensure_ascii=False)

    with open(eval_json.name, 'w', encoding='utf8') as f:
        json.dump(eval_dic, f, ensure_ascii=False)
    
    with open(test_json.name, 'w', encoding='utf8') as f:
        json.dump(test_dic, f, ensure_ascii=False)

    return {"train":train_json.name, "validation":eval_json.name, "test":test_json.name}

def return_class_names(data_files):
    train_file = data_files["train"]
    eval_file = data_files["validation"]
    test_file = data_files["test"]

    class_names = {"pos_tags": set(), "syntax_labels": set()}

    for file in [train_file, eval_file, test_file]:
        with open(file, 'r') as f:
            sentences = f.read().split('\n\n')
            for sentence in sentences:
                lines = sentence.split('\n')
                for line in lines:
                    if line:
                        token, pos_tag, syntax_label = line.split('\t')
                        class_names["pos_tags"].add(pos_tag)
                        if file != test_file:
                            class_names["syntax_labels"].add(syntax_label)

    class_names["pos_tags"] = list(class_names["pos_tags"])
    class_names["syntax_labels"] = list(class_names["syntax_labels"])

    # add UNK label
    class_names["syntax_labels"].append("UNK")

    return class_names

def create_dataset(
        treebank, finetuned, pretrained, encoding, task='single'
    ):
    """
    Create and save an untokenized dataset object from sequence labeling files
    inputs:
        data_dir: directory containing the sequence labeling files
    outputs:
        dataset: a dataset object
    """
    data_dir = 'data/' + encoding + '/' + finetuned + '/' + pretrained + '/' + treebank + '/' + task + '/'

    if encoding == 'const':
        data_files = {
            "train": data_dir + "{}-train.seq_lu".format(treebank),
            "validation": data_dir + "{}-dev.seq_lu".format(treebank),
            "test": data_dir + "{}-test.seq_lu".format(treebank),
        }
    else:
        data_files = {
            "train": data_dir + "train",
            "validation": data_dir + "dev",
            "test": data_dir + "test",
        }

    # Create a json object from the sequence labeling files
    data_json = create_json_files(data_files)

    # Obtain the class names from the sequence labeling files
    class_names = return_class_names(data_files)

    # Create a dataset object from the json object
    dataset = load_dataset(
        "json", 
        data_files=data_json, 
        field="data",
        features=Features(
            {
                "id": Value("int32"),
                "tokens": Sequence(Value("string")),
                "pos_tags": Sequence(ClassLabel(names=class_names["pos_tags"], num_classes=len(class_names["pos_tags"]))),
                "syntax_labels": Sequence(ClassLabel(names=class_names["syntax_labels"], num_classes=len(class_names["syntax_labels"]))),
            }
        ),
    )

    # Save the dataset object
    output_dir = ehd_dir + "datasets" + '/' + encoding + '/' + treebank + '/' + task + '/'
    dataset.save_to_disk(output_dir) # TODO: think about the name of the dataset

    return dataset