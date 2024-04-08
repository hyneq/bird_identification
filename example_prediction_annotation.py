#!/usr/bin/python3

import sys,os

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from bird_identification.image_utils import load_img
from bird_identification.classification.classifier import get_image_classifier
from bird_identification.detection.detector import get_object_detector
from bird_identification.detection_classification import DetectionClassificationResult
from bird_identification.extraction.detection_extraction import extract_detection
from bird_identification.prediction_annotation import get_prediction_annotation

BASE_DIR = os.path.join(os.path.dirname(__file__), "images/example_prediction_annotation")

classifier = get_image_classifier()
detector = get_object_detector()

source_img = cv2.imread(os.path.join(BASE_DIR, "source.png"))

detection_result = detector.predict(source_img)

extracted_img = extract_detection(source_img, detection_result[0])
cv2.imwrite(os.path.join(BASE_DIR, "extracted.png"), extracted_img)

classification_result = classifier.predict(extracted_img)

detection_classification_result = DetectionClassificationResult(detection_result[0], classification_result)

annotated_img = source_img.copy()
get_prediction_annotation([detection_classification_result]).annotate(annotated_img)

cv2.imwrite(os.path.join(BASE_DIR, "annotated.png"), annotated_img)