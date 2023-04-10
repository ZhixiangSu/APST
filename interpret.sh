set -v
dataset="FB15k-237-subset"
suffix="_full"
finding_mode="tail"
seed=42
search_depth=8
support_type=0
support_threshold=0
npaths_ranking=20
device="cuda:0"
nclusters=4


python path_finding_relation.py --dataset $dataset --suffix $suffix --finding_mode $finding_mode --training_mode interpret --npaths_ranking $npaths_ranking --support_threshold $support_threshold --search_depth $search_depth --support_type $support_type

python interpretation.py --device $device --dataset $dataset --suffix $suffix --max_path_num $npaths_ranking --seed $seed --nclusters $nclusters