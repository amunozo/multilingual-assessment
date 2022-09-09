import csv
import os


class InputSLExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a,
                text_a_list,
                text_a_postags=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence.
            label: (Optional) list. The labels for each token. This should be
            specified for training and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = None
        self.text_a_list = text_a_list 
        self.text_a_postags = text_a_postags
        self.labels = labels

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SLProcessor(DataProcessor):
    """Processor for PTB formatted as sequence labeling seq_lu file"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self, data_dir):
        """See base class."""
        
        train_samples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        dev_samples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
        

        train_labels = [label for sample in train_samples 
                            for label in sample.labels]
        
        dev_labels = [label for sample in dev_samples 
                          for label in sample.labels]

        labels = []
        labels.append("[MASK_LABEL]")
        labels.append("-EOS-")
        labels.append("-BOS-")
        train_labels.extend(dev_labels)
        for label in train_labels:
            if label not in labels:
                labels.append(label)
        return labels
    
    
    def _preprocess(self, word):
        if word == "-LRB-": 
            word = "("
        elif word == "-RRB-": 
            word = ")"
        return word

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        sentences_texts = []
        sentences_postags = []
        sentences_labels = []
        sentences_tokens = []
        sentence, sentence_postags, sentence_labels = [],[], []
        tokens = []
        
        for l in lines:
            if l != []:
                
                if l[0] in ["-EOS-","-BOS-"]:
                    tokens.append(l[0])
                    sentence_postags.append(l[-2]) 
                else:     
                    tokens.append(l[0])
                    sentence.append(self._preprocess(l[0]))
                    sentence_labels.append(l[-1].strip())       
                    sentence_postags.append(l[-2]) 
            else:
                
                sentences_texts.append(" ".join(sentence))
                sentences_labels.append(sentence_labels)
                sentences_postags.append(sentence_postags)
                sentences_tokens.append(tokens)
                sentence, sentence_postags, sentence_labels = [], [] ,[]
                tokens = []

        assert(len(sentences_labels), len(sentences_texts))
        assert(len(sentence_postags), len(sentences_texts))
        for guid, (sent, labels) in enumerate(zip(sentences_texts, sentences_labels)):
 
            examples.append(
                InputSLExample(guid=guid, text_a=sent,
                               text_a_list=sentences_tokens[guid],
                               text_a_postags=sentences_postags[guid], 
                               labels=labels))
        return examples