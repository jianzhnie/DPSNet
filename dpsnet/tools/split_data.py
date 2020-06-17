import json
import random


datasets_path = '/data/jianzh/'
f = open('{}instances_test2019.json'.format(datasets_path) ,encoding='utf-8')
gt = json.load(f)

print(gt['info'])
print(gt['licenses'])
print(len(gt['categories']))
print(len(gt['images']))
print(len(gt['annotations']))

## train data
train = dict()
train['info'] = gt['info']
train['licenses'] = gt['licenses']
train['categories'] = gt['categories']
train['images'] = []
train['annotations'] = []


# split the train_data and test_data by 9:1
train_image_size = int(len(gt['images']) * 0.0125)
print('train_img_num:{}'.format(train_image_size))

random.shuffle(gt['images'])
for img_info in gt['images']:
    if len(train['images']) < train_image_size:
        train['images'].append(img_info)
        for anno in gt['annotations']:
            if anno['image_id'] == img_info['id']:
                train['annotations'].append(anno)

with open("/data/jianzh/instances_mini_test2019.json", 'w', encoding='utf-8') as json_file:
    json.dump(train, json_file, indent=4, ensure_ascii=False)




