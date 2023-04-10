import sys
import torch
import numpy as np
from torch.utils.data._utils.collate import default_convert
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, TensorDataset, random_split
from tqdm import tqdm

from colorama import Fore
import os
import random
from utils import load_count_dict, load_cycle, negative_relation_sampling, load_text, load_triplets, \
    myConvert,reshape_relation_prediction_ranking_data,element_wise_cos,cal_metrics
from torch.cuda.amp import GradScaler, autocast
from transformers import set_seed
from models import SentenceTransformer
import torch.nn.functional as F
import argparse
from sklearn import preprocessing
from sklearn.cluster import AffinityPropagation,KMeans,DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


parser = argparse.ArgumentParser(description='Interpretation')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='CUDA device or CPU')
parser.add_argument('--dataset', type=str, default='FB15k-237-subset',
                    help='name of the dataset')
parser.add_argument('--path_dir', type=str, default=None,
                    help='location of extracted paths for each triplet')
parser.add_argument('--text_dir', type=str, default=None,
                    help='location of relation and entity texts')
parser.add_argument('--model_load_file', type=str, default=None,
                    help='location to load pretrained cycle model')
parser.add_argument('--model', type=str, default='sentence-transformers/all-mpnet-base-v2',
                    help='sentence transformer model name on Hugging Face website (huggingface.co)')
parser.add_argument('--tokenizer', type=str, default='sentence-transformers/all-mpnet-base-v2',
                    help='tokenizer name on Hugging Face website (huggingface.co)')
parser.add_argument('--max_path_num', type=int, default=20,
                    help='number of paths loaded for each triplet')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--suffix', type=str, default='_full',
                    help='suffix of the train file name')
parser.add_argument('--output_dir', type=str, default=None,
                    help='directory to store output')
parser.add_argument('--nclusters', type=int, default=4,
                    help='KMeans cluster number')
args = parser.parse_args()

print(args)
set_seed(args.seed)
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device(args.device)

if args.path_dir is None:
    path_dir = os.path.join("data/relation_prediction_path_data/", args.dataset, f'interpret/')
else:
    path_dir=args.path_dir

if args.text_dir is None:
    text_dir = os.path.join("data/data", args.dataset)
else:
    text_dir=args.text_dir

if args.output_dir is None:
    output_dir = os.path.join('data/relation_prediction_path_data/', args.dataset,
                                  f"interpret/")
else:
    output_dir=args.output_dir

if args.model_load_file is None:
    model_load_file = os.path.join('data/relation_prediction_path_data/', args.dataset,
                                   f"interpret/best_val.pth")
else:
    model_load_file=args.model_load_file

interpret_triplets = load_triplets(os.path.join(path_dir, "interpret.txt"))
def load_paths(relation_dir,entity_dir,data_size,max_path_num):
    paths = []
    f1 = open(relation_dir, encoding='utf-8')
    f2 = open(entity_dir, encoding='utf-8')
    for i in range(data_size):
        pnum1=int(f1.readline())
        pnum2=int(f2.readline())
        assert pnum1==pnum2
        paths.append([])
        for j in range(pnum1):
            relations=f1.readline().rstrip('\t\n').split("\t")
            entities=f2.readline().rstrip('\t\n').split("\t")
            if j>=max_path_num:
                continue

            p=[rv for r in zip(entities,relations) for rv in r]
            p.append(entities[-1])
            paths[-1].append(p)
        if pnum1<max_path_num:
            paths[-1].append(["nopath"])
    return paths


interpret_paths = load_paths(os.path.join(path_dir, "relation_paths_interpret.txt"),
                        os.path.join(path_dir, "entity_paths_interpret.txt"), len(interpret_triplets),args.max_path_num)
text,relation_texts = load_text(text_dir)
all_dict = {**text['entity'], **text['relation']}

reshaped_triplets = ["; ".join([all_dict[er] for er in st])+" [SEP]" for st in interpret_triplets]
reshaped_paths = [["; ".join([all_dict[er] for er in s])+" [SEP]" for s in st] for st in interpret_paths]


model=SentenceTransformer(tokenizer_name=args.tokenizer,model_name=args.model,device=device)
model.load_state_dict(torch.load(model_load_file,map_location=device))
model.to(device)

with torch.no_grad():
    triplet_embeds=model(reshaped_triplets).cpu().numpy()
    path_embeds=np.array([model(paths).cpu().numpy() for paths in reshaped_paths])

labels=[]
sim=[]
pbar = tqdm(total=len(triplet_embeds), desc="interpretation",
            position=0, leave=True,
            file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
for i in range(len(triplet_embeds)):
    labels.append(KMeans(n_clusters=args.nclusters).fit_predict(path_embeds[i]))
    sim.append(cosine_similarity(path_embeds[i],np.expand_dims(triplet_embeds[i],0)).squeeze(1))
    dim_reduced=LDA(n_components=2).fit_transform(path_embeds[i],labels[i])
    for j in range(args.nclusters):
        embeds=dim_reduced[labels[-1]==j]
        plt.scatter(embeds[:,0], embeds[:,1], marker='o',label=f'Cluster {j}',s=30)
    plt.rcParams['pdf.fonttype'] = 42
    plt.legend()
    plt.xticks(())
    plt.yticks(())
    plt.rcParams['pdf.fonttype'] = 42
    plt.savefig(os.path.join(output_dir, f"pyfig{i}.eps"), format="eps")
    plt.show()
    pbar.update(1)
pbar.close()

sim = [preprocessing.minmax_scale(sim[i]) for i in range(len(sim))]
with open(os.path.join(output_dir,"result.txt"),"w") as f:
    for i in range(len(reshaped_triplets)):
        f.write(f"{reshaped_triplets[i]}\n")
        f.write(f"{len(reshaped_paths[i])}\n")
        for j in range(len(reshaped_paths[i])):
            f.write(f"{reshaped_paths[i][j]}\t{labels[i][j]}\t{sim[i][j]}\n")
        f.write(f"\n")
