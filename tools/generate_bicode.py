import json
import random

json_file = "./data/thumos/thumos_14_anno.json"
use_random = True
random.seed(42)

with open(json_file, 'r') as f:
    db = json.load(f)

# count annos
anno_cnt = 0
for key, vinfo in db['database'].items():
    annos = vinfo['annotations']
    new_annos = []
    for anno in annos:
        anno_cnt += 1
    
# generate anno numbers
anno_numbers = list(range(anno_cnt))
if use_random: random.shuffle(anno_numbers)

# assign anno number
anno_idx = 0
for key, vinfo in db['database'].items():
    annos = vinfo['annotations']
    new_annos = []
    for anno in annos:
        anno['instance_code'] = anno_numbers[anno_idx]
        anno_idx += 1
        new_annos.append(anno)
    db['database'][key]['annotations'] = new_annos

with open("./data/thumos/thumos_14_bicode_random.json", 'w') as f:
    json.dump(db, f, indent=4)