import pickle

pkl_path = "data/ucf101_2d.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

annotations = data["annotations"]

label_names = {}

for ann in annotations:
    label = int(ann["label"])
    name = ann["frame_dir"].split("_")[1]
    label_names[label] = name

for label, name in sorted(label_names.items()):
    print(label, name)