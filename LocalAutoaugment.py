import random
from PIL import Image
"""
把图片分割成 n×m 的网格。
随机选择 k 个网格。
对每个选中的网格应用一次 AutoAugment 增强。
"""
#
# class LocalizedAutoAugment(object):
#     def __init__(self, autoaugment_policy, n, m, k):
#         """
#         autoaugment_policy: An instance of AutoAugment (e.g., ImageNetPolicy()).
#         n, m: Dimensions of the grid (n rows and m columns).
#         k: Number of grid cells to apply augmentation.
#         """
#         self.autoaugment_policy = autoaugment_policy
#         self.n = n
#         self.m = m
#         self.k = k
#
#     def __call__(self, img):
#         w, h = img.size
#         grid_w, grid_h = w // self.m, h // self.n
#         cells_to_augment = random.sample([(i, j) for i in range(self.n) for j in range(self.m)], self.k)
#
#         for i, j in cells_to_augment:
#             cell = img.crop((j * grid_w, i * grid_h, (j + 1) * grid_w, (i + 1) * grid_h))
#             cell = self.autoaugment_policy(cell)
#             img.paste(cell, (j * grid_w, i * grid_h))
#
#         return img


#####################增加保存图片功能################################
# import random
# from PIL import Image
# import os
#
#
# class LocalizedAutoAugment(object):
#     def __init__(self, autoaugment_policy, n, m, k, save_dir=None):
#         """
#         autoaugment_policy: An instance of AutoAugment (e.g., ImageNetPolicy()).
#         n, m: Dimensions of the grid (n rows and m columns).
#         k: Number of grid cells to apply augmentation.
#         save_dir: Directory to save the augmented images.
#         """
#         self.autoaugment_policy = autoaugment_policy
#         self.n = n
#         self.m = m
#         self.k = k
#         self.save_dir = save_dir
#
#         # Create the directory if it doesn't exist
#         if self.save_dir and not os.path.exists(self.save_dir):
#             os.makedirs(self.save_dir)
#
#     def __call__(self, img):
#         w, h = img.size
#         grid_w, grid_h = w // self.m, h // self.n
#         cells_to_augment = random.sample([(i, j) for i in range(self.n) for j in range(self.m)], self.k)
#
#         for i, j in cells_to_augment:
#             cell = img.crop((j * grid_w, i * grid_h, (j + 1) * grid_w, (i + 1) * grid_h))
#             cell = self.autoaugment_policy(cell)
#             img.paste(cell, (j * grid_w, i * grid_h))
#
#         if self.save_dir:
#             self._save_image(img)
#
#         return img
#
#     def _save_image(self, img):
#         # Generate a unique filename for each saved image
#         file_name = f"augmented_image_{random.randint(1, 1e6)}.jpg"
#         img.save(os.path.join(self.save_dir, file_name))
#####################增加局部概率-全局概率###############################
import random
from PIL import Image
import os

class LocalizedAutoAugment(object):
    def __init__(self, autoaugment_policy, n, m, k, save_dir=None, probability=0.5, whole_image_probability=0.1):
        """
        autoaugment_policy: An instance of AutoAugment (e.g., ImageNetPolicy()).
        n, m: Dimensions of the grid (n rows and m columns).
        k: Number of grid cells to apply augmentation.
        save_dir: Directory to save the augmented images.
        probability: Probability of applying localized augmentation.
        whole_image_probability: Probability of applying augmentation to the whole image.
        """
        self.autoaugment_policy = autoaugment_policy
        self.n = n
        self.m = m
        self.k = k
        self.save_dir = save_dir
        self.probability = probability
        self.whole_image_probability = whole_image_probability

        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __call__(self, img):
        if random.random() < self.whole_image_probability:
            img = self.autoaugment_policy(img)
        elif random.random() < self.probability:
            img = self._apply_localized_augmentation(img)

        if self.save_dir:
            self._save_image(img)

        return img

    def _apply_localized_augmentation(self, img):
        w, h = img.size
        grid_w, grid_h = w // self.m, h // self.n
        cells_to_augment = random.sample([(i, j) for i in range(self.n) for j in range(self.m)], self.k)

        for i, j in cells_to_augment:
            cell = img.crop((j * grid_w, i * grid_h, (j + 1) * grid_w, (i + 1) * grid_h))
            cell = self.autoaugment_policy(cell)
            img.paste(cell, (j * grid_w, i * grid_h))

        return img

    def _save_image(self, img):
        file_name = f"augmented_image_{random.randint(1, 1e6)}.jpg"
        img.save(os.path.join(self.save_dir, file_name))

# 使用示例
# localized_augment = LocalizedAutoAugment(ImageNetPolicy(), n=4, m=4, k=5, save_dir="./localAug/", probability=0.5, whole_image_probability=0.2)
