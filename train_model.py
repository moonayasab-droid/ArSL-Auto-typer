import os
import pickle
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = './ArSL_Data'

data = []
labels = []
skipped_lengths = Counter()

print("Loading training data from folders...")
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):
        for img_path in os.listdir(dir_path):
            if img_path.endswith('.npy'):
                data_path = os.path.join(dir_path, img_path)
                data_array = np.load(data_path, allow_pickle=True)
                flattened = np.array(data_array).flatten()

                if len(flattened) == 126:
                    data.append(flattened)
                    labels.append(dir_)
                else:
                    skipped_lengths[len(flattened)] += 1

# Diagnostics — tells us exactly what's happening with your data
print(f"\nTotal valid samples loaded: {len(data)}")
print("Samples per label:")
for label, count in sorted(Counter(labels).items()):
    print(f"  {label}: {count}")

if skipped_lengths:
    print(f"\nWARNING: Skipped samples with unexpected lengths: {dict(skipped_lengths)}")
    print("These files don't match the expected 126 features and were excluded.\n")

if len(data) == 0:
    print("ERROR: No valid training data found. Check ArSL_Data folder contents.")
    exit()

X = np.asarray(data)
y = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

model = RandomForestClassifier()
print("Training the model...")
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"Model trained successfully! Accuracy: {score * 100:.2f}%")

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print("Saved trained model to 'model.p'") 

