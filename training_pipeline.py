from Utils.data_loader import split_dataset_into_train_test, create_data_loaders
import torch
from torch import nn, optim
from Models.ViT import ViT
from Utils.train import train_pure, train_With_Gradient_Clipping, create_scheduler, train_with_scheduler
from Utils.helper_functions import save_loss_curves, save_results_as_json, save_model
import random

# Set random seeds
torch.manual_seed(42)
random.seed(42)

# Constants
DATASET_PATH = "Data/Garbage classification"
TRAIN_RATIO = 0.8
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 16
IN_CHANNELS = 3
EPOCHS = 15
SEED = 42
MAX_GRAD_NORM = 1.0

# Optimizers to test
OPTIMIZERS = {
    "Adam": lambda params: optim.Adam(params, lr=3e-4),
    "SGD": lambda params: optim.SGD(params, lr=3e-4, momentum=0.9),
    "RMSprop": lambda params: optim.RMSprop(params, lr=3e-4)
}

print("[INFO] Training Pipeline Starting...")
print("********************************************")

print("[INFO] Dataset Splitting into training and test...")
# Dataset Preparation
train_dir, test_dir = split_dataset_into_train_test(dataset_path=DATASET_PATH, 
                                                    train_ratio=TRAIN_RATIO)

print("[INFO] Dataloaders and Class Names are creating...")

train_loader, test_loader, classes = create_data_loaders(train_dir=train_dir, 
                                                         test_dir=test_dir, 
                                                         batch_size=BATCH_SIZE, 
                                                         image_size=IMAGE_SIZE)
print("********************************************")

# Training experiments
experiments = {
    "Pure_Training": train_pure,
    "Gradient_Clipping": train_With_Gradient_Clipping,
    "Scheduler": train_with_scheduler
}

for exp_name, train_function in experiments.items():
    for opt_name, opt_func in OPTIMIZERS.items():
        print(f"[INFO] Running {exp_name} experiment with {opt_name} optimizer...")
        
        # Model Initialization
        model = ViT(img_size=IMAGE_SIZE[0], 
                    patch_size=PATCH_SIZE, 
                    in_channels=IN_CHANNELS, 
                    num_classes=len(classes))
        
        optimizer = opt_func(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Training
        if exp_name == "Pure_Training":
            results = train_pure(
                model=model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                optimizer=optimizer,
                loss_fn=criterion,
                epochs=EPOCHS,
                device=DEVICE
            )
        elif exp_name == "Gradient_Clipping":
            results = train_With_Gradient_Clipping(
                model=model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                optimizer=optimizer,
                loss_fn=criterion,
                epochs=EPOCHS,
                device=DEVICE,
                max_grad_norm=MAX_GRAD_NORM
            )
        elif exp_name == "Scheduler":
            scheduler = create_scheduler(warmup_steps=10000, train_loader=train_loader, optimizer=optimizer)
            results = train_with_scheduler(
                model=model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                optimizer=optimizer,
                loss_fn=criterion,
                epochs=EPOCHS,
                device=DEVICE,
                scheduler=scheduler
            )
        
        # Save results
        save_path = f"Experiments/{exp_name}/{opt_name}"
        save_loss_curves(results, f"{save_path}/accuracy_loss_plot.png")
        save_results_as_json(results, f"{save_path}/training_results.json")
        save_model(model, save_path, f"model_{exp_name.lower()}_{opt_name.lower()}.pth")

        print(f"[INFO] {exp_name} with {opt_name} finished!\n")
