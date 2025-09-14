import os
import json
import random

random.seed(42)

initial_path = '/data/nvme0/aurelius/trainset_300h'
gt_path = '/mnt/blob-data-sigmasystem-out/yuhang/Aurelius/trainset_300h'

json_filename = 'trainset_300h/aurelius_train.json'

with open(json_filename, 'r') as f:
    data = json.load(f)

data_list = list()

for key_name in data.keys():
    if key_name in ['time', 'author']:
        continue
    for subcate in data[key_name].keys():
        for item in data[key_name][subcate]:
            text_prompt = item['text_prompt']
            audio_basename = random.choice(item['reference_audio'])
            assert os.path.exists(os.path.join(initial_path, audio_basename)), f"{audio_basename} does not exist!"

            audio_location = os.path.join(gt_path, audio_basename)

            data_list.append({
                'captions': text_prompt,
                'location': audio_location,
                'duration': 10.0
            })

random.shuffle(data_list)

train_split = int(0.9 * len(data_list))
val_split = int(0.1 * len(data_list))

train_data = data_list[:train_split]
val_data = data_list[train_split:]
test_data = val_data

with open('train.json', 'w') as f:
    json.dump(train_data, f, indent=4)

with open('val.json', 'w') as f:
    json.dump(val_data, f, indent=4)

with open('test.json', 'w') as f:
    json.dump(test_data, f, indent=4)

