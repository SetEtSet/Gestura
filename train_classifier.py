import pickle
import numpy as np
import collections
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data_dict = pickle.load(open('./data.pickle', 'rb'))

one_hand_data, one_hand_labels = [], []
two_hand_data, two_hand_labels = [], []
hand_counts, hand_features = [], []

for sample, label, hand_count in zip(data_dict['data'], data_dict['labels'], data_dict['hand_count']):
    if len(sample) == 42:
        one_hand_data.append(sample)
        one_hand_labels.append(label)
    elif len(sample) == 84:
        two_hand_data.append(sample)
        two_hand_labels.append(label)
    else:
        print(f"Discarding sample with length {len(sample)}")
    hand_counts.append(hand_count)
    hand_features.append([len(sample)])

one_hand_data = np.asarray(one_hand_data)
one_hand_labels = np.asarray(one_hand_labels)
two_hand_data = np.asarray(two_hand_data)
two_hand_labels = np.asarray(two_hand_labels)
hand_counts = np.asarray(hand_counts)
hand_features = np.asarray(hand_features)

def filter_min_samples(data, labels, min_count=2):
    counts = collections.Counter(labels)
    keep_idx = [i for i, lbl in enumerate(labels) if counts[lbl] >= min_count]
    return data[keep_idx], labels[keep_idx]

one_hand_data, one_hand_labels = filter_min_samples(one_hand_data, one_hand_labels)
two_hand_data, two_hand_labels = filter_min_samples(two_hand_data, two_hand_labels)

gesture_model_1h = None
if len(one_hand_data) > 0:
    x_train, x_test, y_train, y_test = train_test_split(
        one_hand_data, one_hand_labels, test_size=0.2, shuffle=True, stratify=one_hand_labels
    )
    gesture_model_1h = RandomForestClassifier(n_estimators=200, random_state=42)
    gesture_model_1h.fit(x_train, y_train)
    y_pred = gesture_model_1h.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print(f'One-hand gestures: {score * 100:.2f}% accuracy')
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cv_scores = cross_val_score(gesture_model_1h, one_hand_data, one_hand_labels, cv=5)
    print(f"Cross-validated accuracy (1-hand): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}")

gesture_model_2h = None
if len(two_hand_data) > 0:
    x_train2, x_test2, y_train2, y_test2 = train_test_split(
        two_hand_data, two_hand_labels, test_size=0.2, shuffle=True, stratify=two_hand_labels
    )
    gesture_model_2h = RandomForestClassifier(n_estimators=200, random_state=42)
    gesture_model_2h.fit(x_train2, y_train2)
    y_pred2 = gesture_model_2h.predict(x_test2)
    score2 = accuracy_score(y_test2, y_pred2)
    print(f'Two-hand gestures: {score2 * 100:.2f}% accuracy')
    print("Confusion Matrix:\n", confusion_matrix(y_test2, y_pred2))
    print("Classification Report:\n", classification_report(y_test2, y_pred2))

    cv_scores2 = cross_val_score(gesture_model_2h, two_hand_data, two_hand_labels, cv=5)
    print(f"Cross-validated accuracy (2-hand): {cv_scores2.mean()*100:.2f}% ± {cv_scores2.std()*100:.2f}")

x_train_h, x_test_h, y_train_h, y_test_h = train_test_split(
    hand_features, hand_counts, test_size=0.2, shuffle=True, stratify=hand_counts
)
hand_model = RandomForestClassifier(n_estimators=200, random_state=42)
hand_model.fit(x_train_h, y_train_h)
y_pred_h = hand_model.predict(x_test_h)
score_h = accuracy_score(y_test_h, y_pred_h)
print(f'Hand-count classification: {score_h * 100:.2f}% accuracy')
print("Confusion Matrix:\n", confusion_matrix(y_test_h, y_pred_h))

with open('model.p', 'wb') as f:
    pickle.dump({
        'gesture_model_1h': gesture_model_1h,
        'gesture_model_2h': gesture_model_2h,
        'hand_model': hand_model
    }, f)
