import os
from preprocess import handle_signalData, data_info
from dotenv import load_dotenv
from utils import SIGNALS
from model import build_model, tuner_random_search, get_best_model

load_dotenv()

MAIN_PATH = os.getenv("MAIN_PATH")
DATA_FOLDER_PATH = f"{MAIN_PATH}/UCI HAR Dataset/UCI HAR Dataset"
os.listdir(DATA_FOLDER_PATH)
TRAIN_FOLDER_PATH = f"{DATA_FOLDER_PATH}/train"
TEST_FOLDER_PATH = f"{DATA_FOLDER_PATH}/test"
os.listdir(TRAIN_FOLDER_PATH)


column_names, activity_labels, activity_labels2 = data_info(DATA_FOLDER_PATH)
x_train, y_train, x_test, y_test = handle_signalData(
    SIGNALS, TEST_FOLDER_PATH, TRAIN_FOLDER_PATH
)

##fine Tune model
best_hp, tuner = tuner_random_search(
    build_model,
    x_train,
    y_train,
    validation_split=0.3,
    trails_count=2,
    objective=["accuracy", "val_accuracy"],
    epochs=50,
)
## get best model hyper parameters
model = get_best_model(tuner, best_hp)

## fit model
history = model.fit(x_train, y_train, validation_split=0.3, epochs=50)
print(
    history["accuracy"],
    history["val_accuracy"],
    history["loss"],
    history["val_loss"],
)
