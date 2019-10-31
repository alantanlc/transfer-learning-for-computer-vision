from __future__ import print_function, division

from torchvision import transforms
import matplotlib.pyplot as plt
from CheXpert import *
from HistogramEqualize import *
from MedianBlur import *

# Directories
root_dir = '/home/alanwuha/Documents/Projects/ce7454-grp17/data/'
csv_dir = '/home/alanwuha/Documents/Projects/ce7454-grp17/data/CheXpert-v1.0-small/'

# Load dataset
image_datasets = {x: CheXpertDataset(csv_file=os.path.join(csv_dir, x + '.csv'), root_dir=root_dir) for x in ['train', 'valid']}

# Individual transforms
mean, std = 127.8989, 74.69
resize = transforms.Resize(365)
randomCrop = transforms.RandomCrop(320)
centerCrop = transforms.CenterCrop(320)
medianBlur = MedianBlur(3)
histEq = HistogramEqualize()
toTensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[mean], std=[std])
toPILImage = transforms.ToPILImage()

# Composed transforms
data_transforms = {
    'train': transforms.Compose([
        resize,
        randomCrop,
        medianBlur,
        histEq,
        toTensor,
        normalize,
        toPILImage
    ]),
    'valid': transforms.Compose([
        resize,
        centerCrop,
        medianBlur,
        histEq,
        toTensor,
        normalize,
        toPILImage
    ])
}

# Apply each of the above train transform on sample
fig = plt.figure()
sample = image_datasets['train'][0]
for i, tsfrm in enumerate([resize, randomCrop, medianBlur, histEq, transforms.Compose([toTensor, normalize, toPILImage]), data_transforms['train']]):
    image = sample['image']
    transformed_sample = tsfrm(image)

    ax = plt.subplot(2, 3, i+1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    plt.imshow(transformed_sample, cmap='gray')

# Apply each of the above train transform on sample
fig = plt.figure()
sample = image_datasets['valid'][0]
for i, tsfrm in enumerate([resize, centerCrop, medianBlur, histEq, transforms.Compose([toTensor, normalize, toPILImage]), data_transforms['valid']]):
    image = sample['image']
    transformed_sample = tsfrm(image)

    ax = plt.subplot(2, 3, i+1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    plt.imshow(transformed_sample, cmap='gray')
plt.show()