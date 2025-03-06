import os

project_name = "DLProject"
os.makedirs(project_name, exist_ok=True)

folders = [
    "configs",
    "dataloader",
    "evaluation",
    "executor",
    "model",
    "notebooks",
    "ops",
    "utils"
]

files = {
    "configs": ["data_config.yaml", "model_config.yaml", "train_config.yaml"],
    "dataloader": ["dataset.py", "transforms.py", "utils.py"],
    "evaluation": ["metrics.py", "visualize.py"],
    "executor": ["train.py", "predict.py", "distributed.py"],
    "model": ["model.py", "loss.py", "optimizer.py"],
    "notebooks": ["exploration.ipynb", "training.ipynb"],
    "ops": ["algebra.py", "image.py", "graph.py"],
    "utils": ["logging.py", "serialization.py", "misc.py"],
    "": ["requirements.txt", "README.md", "main.py"]
}

for folder in folders:
    os.makedirs(os.path.join(project_name, folder), exist_ok=True)

for folder, file_list in files.items():
    for file in file_list:
        with open(os.path.join(project_name, folder, file), "w") as f:
            pass

print("Project structure created successfully!")
