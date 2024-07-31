from typing import Union

import albumentations as albu

TRANSFORM_TYPE = Union[albu.BasicTransform, albu.BaseCompose]
