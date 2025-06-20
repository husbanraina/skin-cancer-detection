import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === Paths ===
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
metadata_path = os.path.join(root_path, '../skin-cancer-detection-dataset/HAM10000_metadata.csv')
part1_path = os.path.join(root_path, '../skin-cancer-detection-dataset/HAM10000_images_part_1')
part2_path = os.path.join(root_path, '../skin-cancer-detection-dataset/HAM10000_images_part_2')
all_images_path = os.path.join(root_path, 'dataset/all_images')

train_path = os.path.join(root_path, 'data/train')
val_path = os.path.join(root_path, 'data/val')

# === Step 1: Merge Images ===
os.makedirs(all_images_path, exist_ok=True)

def copy_all_images(src_dir):
    for file in os.listdir(src_dir):
        if file.endswith('.jpg'):
            src = os.path.join(src_dir, file)
            dst = os.path.join(all_images_path, file)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)

print("Merging image files...")
copy_all_images(part1_path)
copy_all_images(part2_path)
print("✅ All images copied to 'dataset/all_images'")

# === Step 2: Read and Map Metadata ===
df = pd.read_csv(metadata_path)
df = df[df['dx'].isin(['mel', 'nv', 'bkl'])]

label_map = {
    'mel': 'melanoma',
    'nv': 'nevus',
    'bkl': 'seborrheic_keratosis'
}

df['label'] = df['dx'].map(label_map)

# === Step 3: Split Data ===
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# === Step 4: Copy Images into train/val folders ===
def copy_images(df, target_dir):
    for _, row in df.iterrows():
        label = row['label']
        img_id = row['image_id'] + '.jpg'
        src = os.path.join(all_images_path, img_id)
        label_dir = os.path.join(target_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        dst = os.path.join(label_dir, img_id)
        if os.path.exists(src):
            shutil.copyfile(src, dst)

print("Copying training images...")
copy_images(train_df, train_path)

print("Copying validation images...")
copy_images(val_df, val_path)

print("✅ Dataset preparation complete.")
