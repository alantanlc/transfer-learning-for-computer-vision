from __future__ import print_function, division

from torchvision import transforms
import matplotlib.pyplot as plt
from CheXpert import *
from HistogramEqualize import *
from MedianBlur import *
from FrameCrop import *

# Directories
root_dir = '/home/alanwuha/Documents/Projects/ce7454-grp17/data/'
csv_dir = '/home/alanwuha/Documents/Projects/ce7454-grp17/data/CheXpert-v1.0-small/'

# Load dataset
image_datasets = {x: CheXpertDataset(csv_file=os.path.join(csv_dir, x + '.csv'), root_dir=root_dir) for x in ['train', 'valid']}

# Individual transforms
mean, std = 127.8989, 74.69
resize = transforms.Resize(365)
frameCrop = FrameCrop(75)
randomCrop = transforms.RandomCrop(224)
centerCrop = transforms.CenterCrop(224)
medianBlur = MedianBlur(3)
histEq = HistogramEqualize()
toTensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[mean], std=[std])
toPILImage = transforms.ToPILImage()

# Apply each of the above train transform on sample
fig = plt.figure()
titles = ['FrameCrop', 'Resize', 'Random Crop', 'Median Blur', 'Histogram Equalization', 'Normalize', 'CenterCrop', 'Frame Crop + Center Crop']
for j in range(200):
    sample = image_datasets['valid'][j]
    for i, tsfrm in enumerate([transforms.Compose([toPILImage, frameCrop]),
                               transforms.Compose([toPILImage, resize]),
                               transforms.Compose([toPILImage, randomCrop]),
                               transforms.Compose([toPILImage, medianBlur]),
                               transforms.Compose([toPILImage, histEq]),
                               transforms.Compose([toTensor, normalize, toPILImage]),
                               transforms.Compose([toPILImage, resize, centerCrop, medianBlur, histEq]),
                               transforms.Compose([toPILImage, resize, frameCrop, centerCrop, medianBlur, histEq])]):
        image = sample['image']
        transformed_sample = tsfrm(image)

        ax = plt.subplot(3, 3, i+1)
        plt.tight_layout()
        ax.set_title(titles[i])
        plt.imshow(transformed_sample, cmap='gray')
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()