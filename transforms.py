from __future__ import print_function, division

from torchvision import transforms
import matplotlib.pyplot as plt
from CheXpert import *

# Directories
root_dir = '/home/alanwuha/Documents/Projects/ce7454-grp17/data/'
csv_dir = '/home/alanwuha/Documents/Projects/ce7454-grp17/data/CheXpert-v1.0-small/'

# Load dataset
image_datasets = {x: CheXpertDataset(csv_file=os.path.join(csv_dir, x + '.csv'), root_dir=root_dir) for x in ['train', 'valid']}

# Individual transforms
mean, std = 127.8989, 74.69748171138374
resize = transforms.Resize(365)
randomCrop = transforms.RandomCrop(320)
centerCrop = transforms.CenterCrop(320)
toTensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[mean], std=[std])

# Composed transforms
data_transforms = {
    'train': transforms.Compose([
        resize,
        randomCrop,
        toTensor,
        normalize
    ]),
    'valid': transforms.Compose([
        resize,
        centerCrop,
        toTensor,
        normalize
    ])
}

# Apply each of the above train transform on sample
fig = plt.figure()
sample = image_datasets['train'][0]
for i, tsfrm in enumerate([resize, randomCrop, normalize, data_transforms['train']]):
    image = toTensor(sample['image']) if type(tsfrm).__name__ == 'Normalize' else sample['image']
    transformed_sample = transforms.ToPILImage()(tsfrm(image)) if type(tsfrm).__name__ == 'Normalize' else tsfrm(image)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    plt.imshow(transformed_sample)
plt.show()