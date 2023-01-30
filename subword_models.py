import util
import warnings
warnings.simplefilter("ignore", UserWarning)

from transformers import AutoTokenizer, Trainer, TrainingArguments,\
        AutoModelForTokenClassification, DataCollatorForTokenClassification, \
        AutoConfig, AutoModelForMaskedLM
import numpy as np
import os
import evaluate
import datasets
import pandas as pd
import matplotlib.pyplot as plt
import torch

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            new_labels.append(-100)

    return new_labels

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["syntax_labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def train(
        treebank, lm, finetuned, pretrained,
        encoding, task='single', not_ft_lr=2e-3, ft_lr = 5e-5,
        epochs=20,
    ):
    """
    Train a subword token-based model (e.g. BERT, RoBERTa, etc.)
    inputs:
        data_files: dictionary with keys "train", "validation", "test" and values as paths to the corresponding files
        model_name: name of the model to use
        output_dir: path to the directory where the model will be saved
        batch_size: batch size
        epochs: number of epochs
        learning_rate: learning rate
        seed: random seed
    """
    data_dir = 'data/' + encoding + '/' + finetuned + '/' \
         + pretrained + '/' + treebank + '/' + task + '/'
        
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

    
    class_names = util.return_class_names(data_files)  
    if finetuned == 'finetuned':
        lr = ft_lr
    elif finetuned == 'not_finetuned':
        lr = not_ft_lr

    # Datasets are defined by encoding, language/treebank, and task
    dataset_dir = util.ehd_dir
    dataset_dir += 'datasets/' + encoding + '/' + treebank + '/' + task + '/'
    tokenizer = AutoTokenizer.from_pretrained(lm)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    save_dir = util.ehd_dir + 'models/' + encoding + '/' + finetuned + '/' \
         + pretrained + '/' + lm + '/' + treebank + '/' + task + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Initialize log.txt
    log_csv = save_dir + 'log.csv'

    with open(log_csv, 'w') as f:
        f.write("precision,recall,f1,accuracy\n")
    
    label2id = {v: i for i, v in enumerate(class_names["syntax_labels"])}
    id2label = {i: v for i, v in enumerate(class_names["syntax_labels"])}
    num_labels = len(class_names["syntax_labels"])

    if pretrained == 'pretrained':
        model = AutoModelForTokenClassification.from_pretrained(
            lm,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        
        
    elif pretrained == 'not_pretrained':
        # load a model with random weights
        config = AutoConfig.from_pretrained(
            lm,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        model = AutoModelForTokenClassification.from_config(
            config
        )
    
    # Define training arguments
    if treebank == 'UD_Ancient_Greek-Perseus' or treebank == 'chinese':
        per_device_train_batch_size = 16
    else:
        per_device_train_batch_size = 32

    training_args = TrainingArguments(
        save_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=4,
        eval_accumulation_steps = 64,
        evaluation_strategy = "epoch",
        learning_rate=lr,
        num_train_epochs= epochs,
        save_strategy = "no",
        #no_cuda=True
    )

    
    # Load and tokenize the dataset
    dataset = datasets.load_from_disk(dataset_dir)
    dataset = dataset.map(
            tokenize_and_align_labels, 
            batched=True,
            #remove_columns=dataset["train"].column_names,
            fn_kwargs={"tokenizer": tokenizer}
        )
    metric = evaluate.load("seqeval")
    # Define metric
    def compute_metrics(eval_preds):
        print(treebank, lm, finetuned, pretrained, encoding, task, epochs, lr)
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        syntax_features = dataset["train"].features["syntax_labels"]
        label_names = syntax_features.feature.names 
        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        with open(log_csv, 'a') as f:
            f.write(
                str(all_metrics["overall_precision"]) + "," + str(all_metrics["overall_recall"]) + "," + \
                str(all_metrics["overall_f1"]) + "," + str(all_metrics["overall_accuracy"]) + "\n"
            )

        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    if finetuned == 'not_finetuned':
        for name, param in model.named_parameters():
            if name.startswith("bert") or name.startswith("roberta"):
            #if name.startswith("bert.encoder.layer") or name.startswith("roberta.encoder.layer"): # CHECK FOR THE NAME OF THE LAYERS!!
                param.requires_grad = False
        
    # Train the model
    train_result = trainer.train()
    trainer.save_model(save_dir)

    # Plot the results from the log.csv file
    log = pd.read_csv(log_csv)
    log.plot()
    plt.savefig(save_dir + 'log.png')
    plt.close()

def predict(treebank, lm, finetuned, pretrained, encoding, task='single'):
        """
        Predict the test set of a UD treebank using a dependency parsing probe
        and generate a .seq file with the predicted labels
        """
         # Define the variables and paths needed
        # Models
        model_dir = util.ehd_dir + 'models/' + encoding + '/' + finetuned + '/' \
         + pretrained + '/' + lm + '/' + treebank + '/' + task + '/'
        output_dir = model_dir + 'output/'
        output_test = output_dir + 'test.seq'
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(lm)
        data_collator = DataCollatorForTokenClassification(tokenizer)
        # Dataset
        data_dir = 'data/' + encoding + '/' + task + '/' + lm + '/' + treebank + '/'

        dataset_dir = util.ehd_dir + 'datasets/' + encoding + '/' + treebank + '/' + 'single/'
        dataset = datasets.load_from_disk(dataset_dir)
        # Features
        syntax_features = dataset["train"].features["syntax_labels"]
        postag_features = dataset["train"].features["pos_tags"]

        # Define Trainer and TrainingArguments
        training_args = TrainingArguments(
                disable_tqdm=True,
                do_train=False,
                do_eval=False,
                do_predict=True,
                per_device_eval_batch_size=8,
                eval_accumulation_steps = 64,
                output_dir=output_dir
                )
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            eval_dataset=dataset["test"],
        )

        # Tokenize and align the test set
        dataset = dataset.map(
            tokenize_and_align_labels, 
            batched=True,
            #remove_columns=dataset["train"].column_names,
            fn_kwargs={"tokenizer": tokenizer}
        )

        # Define identity dicts
        id2postag = {i: label for i, label in enumerate(postag_features.feature.names)}
        id2label = model.config.id2label

        if treebank != 'german':
            result = trainer.predict(dataset["test"])
            output = result.predictions
            output = np.argmax(output, axis=2)
            outputs = [output]
            results = [result[1]]

        else:
            dataset_test = dataset["test"].train_test_split(test_size=0.5, shuffle=False) # 1/2

            dataset_split = dataset_test["train"].train_test_split(test_size=0.5, shuffle=False)["train"] # train/train
            dataset_used = dataset_split.train_test_split(test_size=0.5, shuffle=False)["train"] # train/train/train
            result_1 = trainer.predict(dataset_used)
            output_1 = result_1.predictions
            result_1 = result_1[1]
            output_1 = np.argmax(output_1, axis=2)
            torch.cuda.empty_cache()
            dataset_used = dataset_split.train_test_split(test_size=0.5, shuffle=False)["test"] # train/train/test
            result_2 = trainer.predict(dataset_used)
            output_2 = result_2.predictions
            result_2 = result_2[1]
            output_2 = np.argmax(output_2, axis=2)
            torch.cuda.empty_cache()

            dataset_split = dataset_test["train"].train_test_split(test_size=0.5, shuffle=False)["test"] # train/test
            dataset_used = dataset_split.train_test_split(test_size=0.5, shuffle=False)["train"] # train/test/train
            result_3 = trainer.predict(dataset_used)
            output_3 = result_3.predictions
            result_3 = result_3[1]
            output_3 = np.argmax(output_3, axis=2)
            torch.cuda.empty_cache()
            dataset_used = dataset_split.train_test_split(test_size=0.5, shuffle=False)["test"] # train/test/test
            result_4 = trainer.predict(dataset_used)
            output_4 = result_4.predictions
            result_4 = result_4[1]
            output_4 = np.argmax(output_4, axis=2)
            torch.cuda.empty_cache()

            dataset_split = dataset_test["test"].train_test_split(test_size=0.5, shuffle=False)["train"] # test/train
            dataset_used = dataset_split.train_test_split(test_size=0.5, shuffle=False)["train"] # test/train/train
            result_5 = trainer.predict(dataset_used)
            output_5 = result_5.predictions
            result_5 = result_5[1]
            output_5 = np.argmax(output_5, axis=2)
            torch.cuda.empty_cache()
            dataset_used = dataset_split.train_test_split(test_size=0.5, shuffle=False)["test"] # test/train/test
            result_6 = trainer.predict(dataset_used)
            output_6 = result_6.predictions
            result_6 = result_6[1]
            output_6 = np.argmax(output_6, axis=2)

            dataset_split = dataset_test["test"].train_test_split(test_size=0.5, shuffle=False)["test"] # test/test
            dataset_used = dataset_split.train_test_split(test_size=0.5, shuffle=False)["train"] # test/test/train
            result_7 = trainer.predict(dataset_used)
            output_7 = result_7.predictions
            result_7 = result_7[1]
            output_7 = np.argmax(output_7, axis=2)
            torch.cuda.empty_cache()
            dataset_used = dataset_split.train_test_split(test_size=0.5, shuffle=False)["test"] # test/test/test
            result_8 = trainer.predict(dataset_used)
            output_8 = result_8.predictions
            result_8 = result_8[1]
            output_8 = np.argmax(output_8, axis=2)

            results = [result_1, result_2, result_3, result_4, result_5, result_6, result_7, result_8]
            outputs = [output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8]

        
        # Get seq file
        tokens = dataset["test"]["tokens"]
        pos_tags = dataset["test"]["pos_tags"]
        pred_syntax_labels = []
        
        for output, result in zip(outputs, results):
            pred_syntax_ids = output
            input_ids = result

            # get aligned predicted syntax labels
            for i in range(len(pred_syntax_ids)):
                sentence_labels = []
                # i is the sentence
                for j in range(len(input_ids[i])):
                    # j is the token
                    input_id = input_ids[i][j]
                    if input_id != -100:
                        label = id2label[output[i][j]]
                        sentence_labels.append(label)
                pred_syntax_labels.append(sentence_labels)

        if len(pred_syntax_labels) != len(tokens):
            raise ValueError("Error: number of sentences in predicted labels is different from number of sentences in test set")
        
        else:
            n_sentences = len(tokens)
            # write file
            file_text = ""
            for i in range(n_sentences):
                sentence_tokens = tokens[i]
                sentence_pos_tags = [id2postag[tag] for tag in pos_tags[i]]
                sentence_syntax_labels = pred_syntax_labels[i]
                sentence_length = len(sentence_tokens)
                for j in range(sentence_length):
                    file_text += '{}\t{}\t{}'.format(sentence_tokens[j], sentence_pos_tags[j], sentence_syntax_labels[j])
                    file_text += '\n'
                file_text += '\n'

            with open(output_test, 'w') as f:
                f.write(file_text)
                    

if __name__ == '__main__':
    languages = ['german'] #, 'basque', 'polish']
    lr = [2e-3, 5e-3]

    for language in languages:
        dataset = util.create_dataset(
            language,
            'bert-base-multilingual-cased',
            '',
            '',
            'spmrl',
            'const')
        for learning_rate in lr:
            train(
                treebank=language,
                lm='bert-base-multilingual-cased',
                finetuned=False,
                pretrained='pretrained',
                encoding='const',
                task='single',
                pr_lr=learning_rate,
                epochs = 10
            )