from Utils.data_loader import split_dataset_into_train_test,create_data_loaders
import torch
from torch import nn
from torch import optim
from Models.ViT import ViT
from Utils.train import train_pure
from Utils.helper_functions import save_loss_curves,save_results_as_json,save_model

DATASET_PATH = "Data/Garbage classification"
TRAIN_RATIO = 0.8
BATCH_SIZE = 32
IMAGE_SIZE = (224,224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 16
IN_CHANNELS = 3
EPOCHS = 10

print("[INFO] Training Pipeline Starting...")

print("********************************************")

print("[INFO] Dataset Splitting into training and test...")

train_dir, test_dir = split_dataset_into_train_test(dataset_path=DATASET_PATH,
                                                    train_ratio=TRAIN_RATIO)

print("********************************************")

print("[INFO] Dataloaders and Class Names are creating...")

train_loader, test_loader, classes = create_data_loaders(
    train_dir=train_dir,
    test_dir=test_dir,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE
)

print("********************************************")

print("1. Experiment : Train ViT by Just Pure Training Loop")
print("[INFO] ViT Model Creating...")

model_pure_training = ViT(img_size=IMAGE_SIZE[0], 
                          patch_size=PATCH_SIZE, 
                          in_channels=IN_CHANNELS, 
                          num_classes=len(classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pure_training.parameters(), lr=3e-4)

print("[INFO] Training Starting...")

result_pure_training = train_pure(model=model_pure_training,
               train_dataloader=train_loader,
               test_dataloader=test_loader,
               optimizer=optimizer,
               loss_fn=criterion,
               epochs=EPOCHS,
               device=DEVICE)

print("[INFO] Training End...")
print("[INFO] Results are saving...")

save_loss_curves(results=result_pure_training,
                 save_path="Experiments/experiment_1/accuracy_loss_plot.png")

save_results_as_json(results=result_pure_training,
                     save_path="Experiments/experiment_1/training_results.json")

save_model(model=model_pure_training,
           target_dir="Experiments/experiment_1",
           model_name="model_pure_training.pth")

print("[INFO] First Experiment Finished..")

print("2. Experiment")
print("Train ViT by Graient Clipping")

print("3. Experiment")
print("Training ViT by LambdaLR scheduler that combine Warmup + Cosine Decay")

