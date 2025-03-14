# Model Rollback Script (rollback_model.py)

import shutil
import datetime

# Load Previous Model Version
with open('models/version.txt', 'r') as file:
    PREVIOUS_MODEL_VERSION = file.read().strip()
CURRENT_MODEL_VERSION = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# Evaluate New Model Accuracy
with open('models/accuracy.txt', 'r') as file:
    NEW_ACCURACY = float(file.read().strip())
with open('models/previous_accuracy.txt', 'r') as file:
    PREVIOUS_ACCURACY = float(file.read().strip())

if NEW_ACCURACY < PREVIOUS_ACCURACY:
    print("New model is underperforming. Rolling back to previous version...")
    shutil.copy(f'models/{PREVIOUS_MODEL_VERSION}/model.pkl', 'models/model.pkl')
    shutil.copy('models/previous_accuracy.txt', 'models/accuracy.txt')
    print("Rollback Completed!")
else:
    print("New model is performing well. Keeping deployment.")
    with open('models/version.txt', 'w') as file:
        file.write(CURRENT_MODEL_VERSION)
    with open('models/previous_accuracy.txt', 'w') as file:
        file.write(str(NEW_ACCURACY))
