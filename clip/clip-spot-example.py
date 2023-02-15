import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Models: {clip.available_models()}")
model, preprocess = clip.load("ViT-B/32", device=device)

img_dir_path = "../spot-images"
categories = ["trash can", "robotics lab", "yummy snack", "something to keep myself warm","happy cloud","place to draw","coffee grounds"]
#categories = ["ukulele"]
category_idx = 0

text = clip.tokenize(categories).to(device)

img_names = os.listdir(img_dir_path)

best_value = 0
best_image = None

for img_name in img_names:
    img_path = img_dir_path + "/" + img_name

    raw_image = Image.open(img_path)
    #raw_image.show()
    image = preprocess(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        #probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        probs = logits_per_image.cpu().numpy()

        if probs[0][category_idx] > best_value:
            best_value = probs[0][category_idx]
            best_image = img_path


    category_probs = [(categories[idx],probs[0][idx]) for idx in range(len(categories))]
    print(f"Categories with probabilities: {category_probs}")

raw_image = Image.open(best_image)
raw_image.show()
