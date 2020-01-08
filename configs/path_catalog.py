import os


# change these variables into the path to the dataset in your environment
coco_dir = os.path.expanduser('~/dataset/coco2017')
voc_dir = os.path.expanduser('~/dataset/voc')

outdir = './out'
logdir = './log'

gdrive_ids = {
    'VOC': {
        'CascadeRCNNResNet50': '',
        'CascadeRCNNResNet101': '',
    },
    'COCO': {
        'CascadeRCNNResNet50': '',
        'CascadeRCNNResNet101': '',
    },
}
