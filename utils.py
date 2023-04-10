
import sys
import torch
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, TensorDataset,random_split
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from colorama import Fore
import os
import random
import copy
import torch.nn.functional as F

def load_triplets_with_pos_tails(path,mode,all_pos_tails,neg_num):
    triplets=[]
    with open(path, encoding='utf-8') as f:
        for i,line in enumerate(f):
            h, r, t = line.split()
            triplets.append([h, r, t])
            if i%neg_num==0:
                all_pos_tails[h][r].append(t)
                all_pos_tails[t][r].append(h)
    return triplets
def load_count_dict(path):
    count_dict={}
    with open(os.path.join(path,"relation_cycle_count.txt"),encoding='utf-8') as f:
        for line in f:
            v,k=line.split(": ")
            k=k.rstrip('\t\n')
            k = " ".join(sorted(k.split("\t")))
            count_dict[k]=float(v)
    return count_dict
def load_triplets(path,mode,all_pos_tails,neg_num):
    triplets=[]
    with open(path, encoding='utf-8') as f:
        for i,line in enumerate(f):
            h, r, t = line.split()
            if mode=='head':
                triplets.append([h, r, t])
            elif mode=='tail':
                triplets.append([t, inverse_relation(r), h])
    return triplets


def load_text(path,long_text,text_length):
    dict={'entity':{},'relation':{},'entitylong':{}}
    long_text_file={
        'GoogleWikipedia':'entity2textGoogleWikipedia.txt',
        'Wikipedia':'entity2textWikipedia.txt',
        'OpenAI':'entity2textOpenAI.txt'
                    }
    relation_texts=[]
    with open(os.path.join(path, "entity2text.txt"),encoding='utf-8') as f1:
        for line in f1:
            entity,text=line.split("\t")
            text=text.rstrip('\n')
            dict['entity'][entity]=text
    with open(os.path.join(path, long_text_file[long_text]),encoding='utf-8') as f2:
        for line in f2:
            entity,text=line.split("\t")
            text=text.rstrip('\n')[:text_length]
            dict['entitylong'][entity]=text

    with open(os.path.join(path, "relation2text.txt"),encoding='utf-8') as f:
        for line in f:
            relation,text=line.split("\t")
            text = text.rstrip('\n')
            words=text.split()
            dict['relation'][relation] = " ".join(words)
            relation_texts.append(" ".join(words))
            words.reverse()
            dict['relation']["{" + relation + "}^-1"] = " ".join(words)
            relation_texts.append(" ".join(words))
    dict['relation']['nopath']=""
    dict['entitylong']['nopath'] = ""
    dict['entity']['nopath'] = ""
    return dict,relation_texts
def load_cycle(path,count_dict=None,count_threshold=10,num_cycles=None):
    relation_cycles=[]
    entity_cycles=[]
    f1=open(os.path.join(path,"relation_cycles.txt"),encoding='utf-8')
    f2=open(os.path.join(path,"entity_cycles.txt"),encoding='utf-8')

    line1_all=f1.readlines()
    line2_all=f2.readlines()
    if num_cycles>0:
        lines=random.sample(list(zip(line1_all,line2_all)),num_cycles)
    else:
        lines=list(zip(line1_all,line2_all))
    for line in lines:
        line1=line[0]
        line2=line[1]
        line1=line1.rstrip('\t\n')
        line2 = line2.rstrip('\t\n')
        if count_dict is None:
            relation_cycles.append(line1.split("\t"))
            entity_cycles.append(line2.split("\t"))
        else:
            k = " ".join(sorted(line1.split("\t")))
            if count_dict[k] > count_threshold:
                relation_cycles.append(line1.split("\t"))
                entity_cycles.append(line2.split("\t"))
    return {'relation':relation_cycles,'entity':entity_cycles}
def inverse_relation(r):
    if r[0]=="{" and r[-4:]=="}^-1":
        return r[1:-4]
    else:
        return "{"+r+"}^-1"


triple2id={'relation':{'2id':{},'id2':[]},'entity':{'2id':{},'id2':[]}}
def tokenize(dataset):
    tokens={'relation':[],'entity':[]}
    for line in dataset['entity']:
        token_line=[]
        for (index,word) in enumerate(line):
            if word not in triple2id['entity']['2id']:
                triple2id['entity']['2id'][word]=len(triple2id['entity']['id2'])
                triple2id['entity']['id2'].append(word)
            token_line.append(triple2id['entity']['2id'][word])
        tokens['entity'].append(token_line)
    for line in dataset['relation']:
        token_line=[]
        for (index,word) in enumerate(line):
            if word not in triple2id['relation']['2id']:
                triple2id['relation']['2id'][word]=len(triple2id['relation']['id2'])
                triple2id['relation']['id2'].append(word)
            token_line.append(triple2id['relation']['2id'][word])
        tokens['relation'].append(token_line)
    # for (index, label) in enumerate(dataset['label']):
    #     if label not in triple2id['relation']['2id']:
    #         triple2id['relation']['2id'][label]=len(triple2id['relation']['id2'])
    #         triple2id['relation']['id2'].append(label)
    #     tokens['label'].append(triple2id['relation']['2id'][label])
    return tokens

def load_neg_relation(path):
    f=open(os.path.join(path,"negative_relations.txt"),encoding="utf-8")
    neg_relations=[]
    for line in f:
        r=line.rstrip('\t\n').split("\t")
        neg_relations.append(r)
    return neg_relations

def negative_relation_sampling(dataset,neg_relations,text,relation_text,sample_num=20):
    d= []
    p_bar = tqdm(total=len(dataset['relation']), desc="negative sampling",
                 position=0, leave=True,
                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    for i in range(len(dataset['relation'])):
        entity=dataset['entity'][i][:-1]
        relation=dataset['relation'][i][1:]
        sentence1 = "".join([text['entity'][e] + "; " + text['relation'][r] + "; "
                              for e,r in zip(entity, relation)
                              ])
        sentence1 =sentence1 + text['entity'][dataset['entity'][i][-1]] +" [SEP]"
        sentence2="".join([
            text['entity'][dataset['entity'][i][0]],"; ",
             text['relation'][inverse_relation(dataset['relation'][i][0])],"; ",
            text['entity'][dataset['entity'][i][-1]]
        ])+" [SEP]"
        sample={'sentence1':[],'sentence2':[],'label':[]}
        sample['sentence1'].append(sentence1)
        sample['sentence2'].append(sentence2)
        sample['label'].append(1)
        for neg in random.sample(neg_relations[i],min(sample_num-1,len(neg_relations[i]))):
            neg_sentence="".join([
            text['entity'][dataset['entity'][i][0]],"; ",
             text['relation'][neg],"; ",
            text['entity'][dataset['entity'][i][-1]]
            ])+" [SEP]"
            sample['sentence1'].append(sentence1)
            sample['sentence2'].append(neg_sentence)
            sample['label'].append(0)

        sl=list(zip(sample['sentence1'],sample['sentence2'],sample['label']))
        random.shuffle(sl)
        sample['sentence1'],sample['sentence2'], sample['label']=zip(*sl)
        d.append([list(sample['sentence1']),list(sample['sentence2']),sample['label'].index(1)])
        p_bar.update(1)
    p_bar.close()
    return d

def sub_relation_sampling(dataset,text):
    d = []
    p_bar = tqdm(total=len(dataset['relation']), desc="sub sampling",
                 position=0, leave=True,
                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    for i in range(len(dataset['relation'])):
        reverse_entity = dataset['entity'][i][:-1]
        reverse_entity.reverse()
        reverse_relation = [inverse_relation(r) for r in dataset['relation'][i][1:]]
        reverse_relation.reverse()
        sentence1 = "".join(["; " + text['relation'][r] + "; " + text['entity'][e]
                             for r, e in zip(reverse_relation, reverse_entity)
                             ])
        sentence1 = text['entity'][dataset['entity'][i][-1]] + sentence1+" [SEP]"
        sentence2 = "".join([
            text['entity'][dataset['entity'][i][-1]], '; ',
            text['relation'][dataset['relation'][i][0]], '; ',
            text['entity'][dataset['entity'][i][0]]
        ])+" [SEP]"
        sample = {'sentence1': [], 'sentence2': [], 'label': []}
        sample['sentence1'].append(sentence1)
        sample['sentence2'].append(sentence2)
        sample['label'].append(1)
        for j in range(1,len(reverse_entity)):
            neg_sentence1 = "".join(["; " + text['relation'][r] + "; " + text['entity'][e]
                                 for r, e in list(zip(reverse_relation, reverse_entity))[:j]
                                 ])
            neg_sentence1 = text['entity'][dataset['entity'][i][-1]] + neg_sentence1+" [SEP]"
            neg_sentence2= "".join([
                text['entity'][dataset['entity'][i][-1]], '; ',
                text['relation'][dataset['relation'][i][0]], '; ',
                text['entity'][dataset['entity'][i][-j-1]]
                ])+" [SEP]"
            sample['sentence1'].append(neg_sentence1)
            sample['sentence2'].append(neg_sentence2)
            sample['label'].append(0)

        sl = list(zip(sample['sentence1'], sample['sentence2'], sample['label']))
        random.shuffle(sl)
        sample['sentence1'], sample['sentence2'], sample['label'] = zip(*sl)
        d.append([list(sample['sentence1']), list(sample['sentence2']), sample['label'].index(1)])
        p_bar.update(1)
    p_bar.close()
    return d


max_seq_length = 128

def convert_cycles_to_features(cycles,embeds,desc):
    p_bar = tqdm(total=len(cycles['label']), desc=desc,
                 position=0, leave=True,
                 file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    input_word_embeds=[]
    input_masks=[]
    input_type_ids=[]
    # for i in range(len(cycles['label'])):
    seq=np.arange(0, len(cycles['label']))
    random.shuffle(seq)
    for i in seq:
        entities=np.array([embeds['entity'][triple2id['entity']['id2'][e]].numpy() for e in cycles['entity'][i]],dtype=np.float)
        relations=np.array([
            embeds['relation'][triple2id['relation']['id2'][e]].numpy()
            if triple2id['relation']['id2'][e] in embeds['relation']
            else embeds['relation'][triple2id['relation']['id2'][e][1:-4]].numpy()
            for e in cycles['relation'][i]],dtype=np.float)
        cycle=np.concatenate([relations,entities],axis=1)
        cycle=cycle.reshape(-1,cycle.shape[2])
        cycle=np.concatenate([embeds['special']['[CLS]'].numpy(),cycle[-1:],cycle[0:2],embeds['special'][';'].numpy(),cycle[1:],embeds['special']['[CLS]'].numpy()],axis=0)

        input_type=np.zeros(len(cycle))
        input_type[5:]=1
        input_mask=np.ones(cycle.shape[0],dtype=np.float64)

        cycle=np.pad(cycle,[[0,max_seq_length-len(cycle)],[0,0]],mode='constant')
        input_mask = np.pad(input_mask, [0, max_seq_length - len(input_mask)], mode='constant')
        input_type =np.pad(input_type, [0, max_seq_length - len(input_type)], mode='constant')
        input_word_embeds.append(cycle)
        input_type_ids.append(input_type)
        input_masks.append(input_mask)
        p_bar.update(1)

    labels=np.array(cycles['label'],dtype=np.int64)
    p_bar.close()



    return [torch.tensor(input_word_embeds, dtype=torch.float),
            torch.tensor(input_masks, dtype=torch.float),
            torch.tensor(input_type_ids, dtype=torch.int64),
            torch.tensor(labels, dtype=torch.int64)]


def load_paths(relation_dir,head_dir,data_size,max_path_num,mode):
    paths = []
    f1 = open(relation_dir, encoding='utf-8')
    f2 = open(head_dir, encoding='utf-8')
    for i in range(data_size):
        pnum1=int(f1.readline())
        pnum2=int(f2.readline())
        assert pnum1==pnum2
        paths.append([])
        for j in range(pnum1):
            relations=f1.readline().rstrip('\t\n').split("\t")
            entities=f2.readline().rstrip('\t\n').split("\t")
            if j >= max_path_num:
                continue

            p = [rv for r in zip(entities, relations) for rv in r]
            p.append(entities[-1])
            paths[-1].append(p)
        for i in range(pnum1,max_path_num):
            paths[-1].append(["nopath"])
    return paths

# def load_rules(relation_dir,head_dir,data_size,max_path_num):
#     paths = []
#     f1 = open(relation_dir, encoding='utf-8')
#     f2 = open(head_dir, encoding='utf-8')
#     for i in range(data_size):
#         pnum1=int(f1.readline())
#         pnum2=int(f2.readline())
#         assert pnum1==pnum2
#         paths.append([])
#         for j in range(pnum1):
#             relations=f1.readline().rstrip('\t\n').split("\t")
#             heads=f2.readline().rstrip('\t\n').split("\t")
#             head=heads[0]
#             if j>=max_path_num:
#                 continue
#             p=[head,relations]
#             paths[-1].append(p)
#         # for i in range(pnum1,max_path_num):
#         #     paths[-1].append(["nodesc",["nopath"]])
#     return paths
# def merge_rules_and_paths(paths,rules,neg_num,max_path_num):
#     merged = copy.deepcopy(paths)
#     # for i, p in enumerate(paths):
#     #     for j in range(0, min(max_path_num - len(p), len(rules[i]))):
#     #         merged[i].append(rules[i][j])
#     for m in merged:
#         for i in range(len(m),max_path_num):
#             m.append(["nopath"])
#     return merged
def reshape_relation_prediction_ranking_data(ranking_triplets,ranking_paths,neg_size,all_dict):
    reshaped_triplets=[]
    reshaped_paths=[]
    labels=[]
    indexes=[]
    for i in range(0,len(ranking_triplets),neg_size):
        triplets=ranking_triplets[i:i+neg_size]
        paths=ranking_paths[i:i+neg_size]
        inverse_index=list(np.arange(neg_size))
        truth=triplets[0]
        t_p=list(zip(triplets,paths,inverse_index))
        random.shuffle(t_p)
        triplets,paths,inverse_index=zip(*t_p)
        paths=list(paths)
        label=triplets.index(truth)
        reshaped_triplets.append(triplets)
        reshaped_paths.append(paths)
        labels.append(label)
        index=np.argsort(inverse_index)
        indexes.append(index)
    # reshaped_triplets = [[f'{all_dict[st1[1]]} [SEP] {all_dict[st1[0]]} [SEP] {all_dict[st1[2]]}' for st1 in st2] for st2 in reshaped_triplets]
    # reshaped_paths = [[[f'{"; ".join([all_dict[r] for r in st[1]])} [SEP] {all_dict[st[0]]} [SEP] {all_dict[st[2]]}' for st in st1] for st1 in st2] for st2 in reshaped_paths]
    return reshaped_triplets,reshaped_paths,labels,indexes

def myConvert(data):
    for i in range(len(data[0])):
        yield [d[i] for d in data]

def element_wise_cos(a,b):
    a_norm=torch.sqrt(torch.sum(torch.square(a),dim=1)).unsqueeze(0)
    b_norm = torch.sqrt(torch.sum(torch.square(b), dim=1)).unsqueeze(0)
    return torch.matmul(b,a.T) / torch.matmul(b_norm.T,a_norm)

# def reshape_link_prediction_ranking_data(ranking_triplets,ranking_paths,mode,training):
#     labels = []
#     reshaped_triplets = []
#     reshaped_paths=[]
#
#     if mode=='head':
#         target=-1
#     else:
#         target=0
#     for i in range(len(ranking_triplets)):
#         target_list= []
#         path=[]
#         for line in ranking_paths[i]:
#             if line[target] not in target_list:
#                 target_list.append(line[target])
#                 path.append([])
#             path[target_list.index(line[target])].append(line)
#         if ranking_triplets[i][target] not in target_list:
#             if training=='train':
#                 continue
#             else:
#                 labels.append(-1)
#                 reshaped_paths.append(path)
#         else:
#             labels.append(target_list.index(ranking_triplets[i][target]))
#             reshaped_paths.append(path)
#         reshaped_triplets.append([ranking_triplets[i].copy() for j in range(len(target_list))])
#         for j in range(len(target_list)):
#             reshaped_triplets[-1][j][target]=target_list[j]
#
#     return reshaped_triplets,reshaped_paths,labels
def cal_metrics(predictions,labels,pos_mask):
    predictions[pos_mask==True]=0
    predictions=np.array(predictions)
    labels=np.array(labels)
    sorted_index=np.argsort(-predictions,axis=1)
    pos=np.array([np.where(sorted_index[i]==labels[i]) for i in range(len(labels))]).flatten()
    hit1 = len(pos[pos<1])/len(pos)
    hit3 = len(pos[pos < 3])/len(pos)
    hit10 = len(pos[pos < 10])/len(pos)
    MR=np.average(pos+1)
    MRR=np.average(1/(pos+1))
    return [MR,MRR,hit1,hit3,hit10]

def CosineEmbeddingLoss(outputs,labels,margin=0.):
    scores=outputs.permute(2,0,1)
    y=F.one_hot(labels,num_classes=outputs.shape[1]).repeat(outputs.shape[2],1,1)
    pos_loss=(1-scores)*y
    neg_loss=(1-y)*torch.clamp(scores-margin,min=0)
    loss=torch.sum(pos_loss+neg_loss)/outputs.shape[2]
    return loss

