from argparse import ArgumentParser
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter, OrderedDict
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import copy
import os
import pickle
from _curses import tparm
import chardet



class WordConll(object):
    
    def __init__(self, idx, word, head, deprel):
        self.idx = idx
        self.word = word
        self.head = head
        self.deprel = deprel


def dependency_displacements(tree, unlabeled=False):
    
    distances = []
    for word in tree[1:]:
        if word.head != 0:
            distance = word.idx - word.head
            if unlabeled:
                distances.append(distance)
            else:
                distances.append(str(distance)+"_"+word.deprel)
    # include only those that appear more than 5 times
    return distances
    

def word_relations(tree, unlabeled=True):
    
    relations = []
    for word in tree[1:]:
        if unlabeled:
            relations.append(word.deprel)
        else:
            relations.append(str(word.head)+"_"+word.deprel)
    
    return relations
       


def model2legends(model_name):
    
    if model_name.lower().startswith("elmo"):
        return "ELMo"
    elif model_name.lower().startswith("bert"):
        return "mbert"
    elif model_name.lower().startswith("random"):
        return "Random"
    elif model_name.lower().startswith("glove"):
        return "GloVe"
    elif model_name.lower().startswith("wiki-news"):
        return "FastText"
    elif model_name.lower().startswith("google"):
        return "Word2vec"
    elif model_name.lower().startswith("dyer"):
        return "S. word2vec"
    else:
        return model_name


def read_conllu(conllu_sentence):
    

        lines = conllu_sentence.split("\n")
        tree = [WordConll(0,"-ROOT-","_","_")]
        for l in lines:
            if l.startswith("#"):continue
            if l != "":
                ls = l.split("\t")
                if "." in ls[0] or "-" in ls[0]: continue
                word_conllu= WordConll(int(ls[0]), ls[1], int(ls[6]), ls[7])
                tree.append(word_conllu)
        
        return tree

        

def elements_to_plot_from_conll(path_file, unlabeled):    
    
    with open(path_file, 'r') as f:
        print("Reading file: {}".format(path_file.split("/")[-1]))
        sentences = f.read().split("\n\n")
        distances = []
        relations = []
        trees = []
        for s in sentences[:-1]:
            tree = read_conllu(s) 
            trees.append(tree)
            distances.extend(dependency_displacements(tree, unlabeled))
            relations.extend(word_relations(tree, unlabeled))    
    return distances, relations


def dependency_head_performance(gold_labels, pred_labels):
 
    relations = {}
    print("Length of gold_labels: {}".format(len(gold_labels)))
    print("Length of pred_labels: {}".format(len(pred_labels)))
    if len(gold_labels) != len(pred_labels):
        raise ValueError("Length of gold_labels and pred_labels do not match")
    assert len(gold_labels) == len(pred_labels)
    for gold_element, pred_element in zip(gold_labels, pred_labels):
     
        gold_head = int(gold_element.split("_")[0]) 
        gold_relation =  gold_element.split("_")[1] 
        pred_head = int(pred_element.split("_")[0]) 
        pred_relation = pred_element.split("_")[1]
 
        if gold_relation not in relations:       
            relations[gold_relation] = {"total_desired":0.,"total_predicted":0.,"correct":0.}
 
        if pred_relation not in relations:
            relations[pred_relation] = {"total_desired":0.,"total_predicted":0.,"correct":0.}
         
        relations[gold_relation]["total_desired"]+=1
        relations[pred_relation]["total_predicted"]+=1
         
        if gold_element == pred_element:
            relations[gold_relation]["correct"]+=1
         
    scores ={}
    for rel in relations:
        desired = relations[rel]["total_desired"];
        predicted = relations[rel]["total_predicted"]
        correct = relations[rel]["correct"]
        p = 0 if predicted == 0 else correct / predicted
        r = 0 if desired == 0 else correct / desired
        scores[rel] = {"p": p, "r": r}

    return scores 


def displacement_labeled_performance(gold_distances, pred_distances):
    
    
    distances = {}
    if len(gold_distances) != len(pred_distances):
        raise ValueError("Length of gold_distances and pred_distances do not match")
    
    assert len(gold_distances) == len(pred_distances), "Lengths of gold_distances and pred_distances do not match"
    for gold_element, pred_element in zip(gold_distances, pred_distances):
        if gold_distances.count(gold_element) < 10:
            continue
        gold_distance = int(gold_element.split("_")[0]) 
        pred_distance = int(pred_element.split("_")[0]) 
        
        if gold_distance not in distances:       
            distances[gold_distance] = {"total_desired":0.,"total_predicted":0.,"correct":0.}
 
        if pred_distance not in distances:
            distances[pred_distance] = {"total_desired":0.,"total_predicted":0.,"correct":0.}
         
        distances[gold_distance]["total_desired"]+=1
        distances[pred_distance]["total_predicted"]+=1
         
        if gold_element == pred_element:
            distances[gold_distance]["correct"]+=1

    #Computing precision and recall

    scores ={}
    for d in distances:
        desired = distances[d]["total_desired"];
        predicted = distances[d]["total_predicted"]
        correct = distances[d]["correct"]
        p = 0 if predicted == 0 else correct / predicted
        r = 0 if desired == 0 else correct / desired
        scores[d] = {"p": p, "r": r}
 
    return scores 
    



if __name__ == '__main__':
    
    arg_parser = ArgumentParser()

    arg_parser.add_argument("--predicted", 
                            help="Path to the directory containing the predicted input files in conllu format", 
                            default=None)
    arg_parser.add_argument("--gold",
                            help="Path to the gold file in conll format")
    arg_parser.add_argument("--unlabeled", 
                            default=False,
                            action="store_true",
                            help="Ignores dependency types")
    
    arg_parser.add_argument("--output",
                            help="Path to the output .png file")
    
    args = arg_parser.parse_args()
    
    ############################################################################
    # Computing:
    # 1. Dependency displacements for the gold trees (gold_distances)
    # 2. Gold dependency relations (no head index taken into account)
    ############################################################################    
    all_distances = []
    idx_distances = OrderedDict()

    gold_distances, gold_relations = elements_to_plot_from_conll(args.gold, args.unlabeled)         
    gold_relations_counter = Counter(gold_relations)

    input_files = sorted([args.predicted+os.sep+f 
                   for f in os.listdir(args.predicted)])    

    #Variables to compute the dependency displacement scores
    distances = []
    distances_precision  = []
    distances_recall = []
    distances_f1 = []
    distances_models = []
    
    #Variables to compute the performance on the most common dependency relations
    relations = []
    relations_precision = []
    relations_recall = []
    relations_f1 = []
    relations_models = []
    relations_occ = []

    for input_file in input_files:        
    
        pred_distances, pred_relations = elements_to_plot_from_conll(input_file, args.unlabeled)
        model_name = input_file.rsplit("/",1)[1]
        model_name = model_name.replace(".test.outputs.txt","")   
        print ("Processing file:", input_file)
        ########################################################################
        # Performance on different dependency relations 
        # (not penalizing the index of the head)
        ########################################################################
        rel2idx = {e:idx for idx,e in enumerate(sorted(set(gold_relations).union(set(pred_relations))))}
        idx2rel = {rel2idx[e]:e for e in rel2idx}
        
        if args.unlabeled:
            indexed_gold_relations = [rel2idx[g] for g in gold_relations]
            indexed_pred_relations = [rel2idx[p] for p in pred_relations]
            precision, recall, f1, support = precision_recall_fscore_support(indexed_gold_relations, indexed_pred_relations)
            
            for relation, occurrences in gold_relations_counter.most_common(n=7):
                
                idx_relation = rel2idx[relation]
                relations.append(relation)
                idx_relation = int(idx_relation)
                relations_precision.append(precision[idx_relation]) # type: ignore
                relations_recall.append(recall[idx_relation]) # type: ignore
                relations_f1.append(f1[idx_relation]) # type: ignore
                relations_models.append(model2legends(model_name))

        else:

            relations_performance = dependency_head_performance(gold_relations, pred_relations)             
            _, aux_gold_relations = elements_to_plot_from_conll(args.gold, True)         
            aux_gold_relations_counter = Counter(aux_gold_relations)
            f1_score = lambda p,r : 0 if p== 0 and r == 0 else round(2*(p*r) / (p+r),4)
            
            for relation, occurrences in aux_gold_relations_counter.most_common(n=7): 
                relations.append(relation)
                relations_occ.append(occurrences)
                relation_p = relations_performance[relation]["p"]
                relation_r = relations_performance[relation]["r"]
                relations_precision.append(relation_p)
                relations_recall.append(relation_r)
                relations_f1.append(f1_score(relation_p, relation_r))
                relations_models.append(model2legends(model_name))
    
        #########################################################################
        # Performance for each displacement, labeled or unlabeled
        #########################################################################
        if args.unlabeled:        
            labelsi = {e:idx for idx,e in enumerate(sorted(set(gold_distances).union(set(pred_distances))))}
            ilabels = {labelsi[e]:e for e in labelsi}
            aux_gold_distances = [labelsi[g] for g in gold_distances]
            aux_pred_distances = [labelsi[p] for p in pred_distances]
            precision, recall, f1, support = precision_recall_fscore_support(aux_gold_distances, aux_pred_distances)
            
        else:
            
            distances_performance = displacement_labeled_performance(gold_distances, pred_distances)
            precision =[]
            recall = []
            f1 = []
            support = []
            f1_score = lambda p,r : 0 if p== 0 and r == 0 else round(2*(p*r) / (p+r),4)
            
            labelsi = {e:idx for idx,e in enumerate(  sorted(set(list(map(int,distances_performance)))  ))}
            ilabels = {labelsi[e]:e for e in labelsi}
            
            for distance in sorted(list(map(int,distances_performance))):    
                distance_p = distances_performance[distance]["p"]
                distance_r = distances_performance[distance]["r"]
                precision.append(distance_p)
                recall.append(distance_r)
                f1.append(f1_score(distance_p, distance_r))
                support.append(None)
            
            
        for idxe, (p,r,f,s) in enumerate(zip(precision,recall,f1,support)): # type: ignore

            if abs(int(ilabels[idxe])) <= 20:
                distances.append(ilabels[idxe])
                distances_precision.append(p)
                distances_recall.append(r)
                distances_f1.append(f)
                distances_models.append(model2legends(model_name))
        
    d = {"distances": distances,
             "precision": distances_precision,
             "recall": distances_recall,
             "f1-score": distances_f1,
             "model": distances_models}

    
       
    data = pd.DataFrame(d)
    data = data[data['model'] != 'gold']

    ############################################################################
    #                 PLOTTING DEPENDENCY DISPLACEMENTS                        #
    ############################################################################
            
    markers = ['o','.',',','*','v','D','h','X','d','^','o','o','<','>']
    sns.set(style="whitegrid")
    # enlarge all fonts and marks

    sns.set_context("paper", font_scale=1.8, rc={"lines.linewidth": 2, 'lines.markersize': 5})
    
    #palette = dict(zip(sorted(set(distances_models)), sns.color_palette()))
    palette = 'Set2'
    
    ax = sns.lineplot(x="distances", y="f1-score",
                 hue="model", style="model",
                 data=data,
                 palette=palette,
                 linewidth=2.5,
                 markersize=8,
                 markers=True,
                 dashes=False,
                 )


    handles, labels = ax.get_legend_handles_labels()
    # delete legend
    #ax.legend_.remove()

    #ax.tick_params(labelsize=30)
    ax.set_xlabel("Dependency displacement")#,fontsize=35)
    ax.set_ylabel("F1-score")#,fontsize=35)
    # set y-axis limits 0-1
    ax.set_ylim(0,1)
    ax.tick_params(labelsize=20)
    # larger x- and y- labels titles
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)

    plt.savefig(args.output, bbox_inches='tight', dpi=300)
    
    # linear legend outside the plot
   # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    #plt.show()
    
    ############################################################################
    #                PLOTTING F1-SCORE/DEPENDENCY-RELATIONS                    #
    ############################################################################
    # new fig
    plt.figure()
    d_rel = {"relations": relations,
             "relations_occ": relations_occ,
             "precision": relations_precision,
             "recall": relations_recall,
             "f1-score": relations_f1,
             "model": relations_models}
    
    data_rel = pd.DataFrame(d_rel)    
    data_rel = data_rel[data_rel['model'] != 'gold']
    
    data_rel.sort_values(by=['model','relations_occ'],  inplace=True, ascending=[False,False]) # it was [0,0] before but Copilot suggested [False,False]

    ax.set_ylim(0,1)
    # plot order: punct, nmod, advmod, nsubj, root, obj, case
    order = ['punct','nmod','advmod','nsubj','root','obj','case']

    ax = sns.barplot(x="relations", y="f1-score",
                 hue="model", 
                 data=data_rel,
                 palette=palette,
                 order=order,)

    handles, labels = ax.get_legend_handles_labels()
    labels = ["("+str(idl)+") "+l for idl,l in enumerate(labels,1)] 

    plt.setp(ax.get_legend().get_texts()) # for legend text
    
    ax.set_xlabel("Dependency relation")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0,1)
    plt.savefig(args.output.replace('displacement', 'relation'), bbox_inches='tight', dpi=300)

    #plt.show()