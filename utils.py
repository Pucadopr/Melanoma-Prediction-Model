from collections import Counter
from tqdm import tqdm
from typing import Optional, List

# possible reference: https://www.programiz.com/python-programming/methods/string/join

def get_file_type(image_folder_path: str, allowed_extensions: Optional[List]=None):
    if allowed_extensions is None:
        allowed_extensions = ['.jpg', '.png', '.jpeg']

    extension_type = []
    file_list = os.listdir(image_folder_path)
    for file in tqdm(file_list):
        extension_type.append(os.path.splitext(file)[-1].lower())
    
    extension_dict = Counter(extension_type)
    assert len(extension_dict.keys()) == 1, "The extension in the folder should all be the same, but found {} extensions".format(extension_dict.keys)
    extension_type = list(extension_dict.keys())[0]
    assert extension_type in allowed_extensions
    return extension_type

def get_transforms(config):
    transforms_train = albumentations.Compose(
        [

            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightness(limit=0.2, p=0.75),
            albumentations.RandomContrast(limit=0.2, p=0.75),

            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            albumentations.Resize(height=config.image_size, width=config.image_size, p=1.0),
            # Test yourself on whether doing cutout last affects the seqeunce order?

            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    transforms_val = albumentations.Compose(
        [
            albumentations.Resize(height=config.image_size, width=config.image_size, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    return transforms_train, transforms_val

class AverageLossMeter:
    """
    Computes and stores the average and current loss
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.curr_batch_avg_loss = 0
        self.avg = 0
        self.running_total_loss = 0
        self.count = 0

    def update(self, curr_batch_avg_loss: float, batch_size: str):
        self.curr_batch_avg_loss = curr_batch_avg_loss
        self.running_total_loss += curr_batch_avg_loss * batch_size
        self.count += batch_size
        self.avg = self.running_total_loss / self.count


# Maybe compare with utils.py from source
class AccuracyMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.score = 0
        self.count = 0
        self.sum = 0

    def update(self, y_true, y_pred, batch_size=1):

        # so we just need to count total num of images / batch_size
        # self.count += num_steps
        self.batch_size = batch_size
        self.count += self.batch_size
        # this part here already got an acc score for the 4 images, so no need divide batch size
        self.score = sklearn.metrics.accuracy_score(y_true, y_pred)
        total_score = self.score * self.batch_size

        self.sum += total_score

    # 1. I doubt I need to use @property here, but I saw one guy used it, so I am confused.
    @property
    def avg(self):
        self.avg_score = self.sum / self.count
        return self.avg_score