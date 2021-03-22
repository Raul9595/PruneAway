import glob
from dataloader import LoadImages
from label_mappings import labels
from tqdm import tqdm
import torch

def eval_data(model, file, param_str):
    imagenet_path = '/data/datasets/community/deeplearning/imagenet/val/'

    # Creating a dictionary of label files
    label_dict = {}
    with open("val.txt") as f:
        for line in f:
            (key, val) = line.split()
            label_dict[key] = val

    dataset = LoadImages(imagenet_path)
    correct = 0
    total = 0
    wrong = 0

    model.eval()

    file = open(file, "w")
    file.write(param_str + '\n\n')
    pf = '%50s' + '%40s'
    file.write(pf % ('Path', 'Result') + ' \n')

    print(('\n' + '%10s' * 3) % ('Corect', 'Wrong', 'Total'))

    with tqdm(total=50000) as pbar:
        for path, img, im0s in dataset:
            flag = 1
            out = model(img)

            _, index = torch.max(out, 1)

            if int(label_dict[path.split('/')[-1]]) == index[0].item():
                correct+=1
            else:
                wrong+=1
                flag = 0
            total+=1

            s = ('%10.4g' * 3) % (correct, wrong, total)
            pbar.set_description(s)

            pf = '%50s' + '%10.3g' * 1 # print format
            file.write(pf % (path, flag) + ' \n')