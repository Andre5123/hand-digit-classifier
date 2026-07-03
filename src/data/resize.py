# This script resizes the detection training images.
import shutil
import cv2

from pathlib import Path
img_paths = list(Path("../../data/custom/images").glob("*.jpg"))
backup_dir = Path("../../data/custom/images_original")

new_size = (416,416)

for img_path in img_paths:
    shutil.copy(str(img_path), str(backup_dir / img_path.name))
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, new_size)
    cv2.imwrite(str(img_path), img)