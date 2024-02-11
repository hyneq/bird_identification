#!/usr/bin/python3

import sys, os, glob
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from bird_identification.image_utils import load_img

from bird_identification.classification.classifier import ImageClassifier, get_image_classifier

classifier: ImageClassifier = get_image_classifier(input_strategy=load_img)

INPUT_DIR = sys.argv[1]

results = {}
dircetory_glob = glob.glob(os.path.join(INPUT_DIR, "*"))
for dir_path in dircetory_glob:
    expected_class = os.path.basename(dir_path)
    results[expected_class] = results_current = {"total": 0, "correct": 0, "correctness": 0.0}

    img_glob = glob.glob(os.path.join(dir_path, "*.*"))
    for img_path in img_glob:
        classification_result = classifier.predict(img_path)

        if classification_result.class_name == expected_class:
            results_current["correct"] += 1
        else:
            print(f"Image from {img_path} does does not match with class {expected_class}: identified as {classification_result.class_name}")
        
        results_current["total"] += 1
    
    if results_current["total"]:
        results_current["correctness"] = results_current["correct"] / results_current["total"]


print()
print("Correctness for each class:")

for class_name, result in results.items():
    print(f"{class_name}: {result['correct']}/{result['total']} {result['correctness']*100:.2f}%")

print()

overall_total = sum(result['total'] for result in results.values())
overall_correct = sum(result['correct'] for result in results.values())
overall_correctness = overall_correct / overall_total

print(f"Overall correctness: {overall_correct}/{overall_total} {overall_correctness*100:.2f}%")