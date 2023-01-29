#!/usr/bin/python3

from detection import *
from classification import get_image_classifier, classify_images

def extract_detected(images, label, box_expand=[0.1, 0.1], model=None, save_path=None, i=None, confidence=0.55, i_placeholder="$i", class_placeholder="$c", i_padding=5, return_array=False, verbose=True, classify=None, classification_model=None, classification_confidence=0.85):
    if type(images) is str:
        images = list(sorted(glob.glob(images)))
    elif type(images[0]) is str:
        images_globbed = []
        for image in images:
            if type(image) is str:
                images_globbed += list(glob.glob(images))
        
        images = images_globbed
    
    detector = get_object_detector(model=model, classes=label, min_confidence=confidence)

    if classify == None:
        if save_path:
            classify = (class_placeholder in save_path)
        else:
            classify = False

    if classify:
        classifier = get_image_classifier(model=classification_model, min_confidence=classification_confidence)

        class_names = classifier.model.class_names.class_names + ["__not_recognized__"]

        if save_path:
            save_path_classes = {}
            i = {}
            for class_name in class_names:
                save_path_classes[class_name] = save_path.replace(class_placeholder, class_name)
                i[class_name] = get_i(save_path_classes[class_name])
        
        extracted_images = {}

        if return_array:
            for class_name in class_names:
                extracted_images[class_name] = []

    else:
        if save_path and not i: 
            i = get_i(save_path)

        if return_array:
            extracted_images = []

    for image in images:
        if type(image) == str:
            im_path = image
            image = cv2.imread(im_path)
        else:
            im_path = None
        
        results = detect_objects(image, detector=detector)[0]
        for result in results:

            box = result.bounding_box

            x_expand = int(box.width*box_expand[0])
            x_start = max(box.x-x_expand,0)
            x_end = min(box.x+box.width+x_expand, image.shape[1])
            x = slice(x_start, x_end)

            y_expand = int(box.height*box_expand[1])
            y_start = max(box.y-y_expand,0)
            y_end = min(box.y+box.height+y_expand, image.shape[0])
            y = slice(y_start, y_end)

            extracted_image = image[y, x]

            if verbose:
                print("Extracted image from {} at {} with {} % ".format(im_path, box, np.round(result.confidence[0]*100,2)))
            
            if classify:
                classification_result = classify_images(extracted_image, classifier=classifier)[0]
                if len(classification_result.class_names):
                    class_name = classification_result.class_names[0]
                else:
                    class_name = "__not_recognized__"

            if return_array:
                if classify:
                    extracted_images[class_name].append(extracted_image)
                else:
                    extracted_images.append(extracted_image)

            if save_path:
                if classify:
                    extracted_image_path = save_path_classes[class_name].replace(i_placeholder, str(i[class_name]).zfill(i_padding))
                    i[class_name] += 1
                else:
                    extracted_image_path = save_path.replace(i_placeholder, str(i).zfill(i_padding))
                    i += 1
                
                if verbose:
                    print("Saving image from {} at {} to {}".format(im_path, box, extracted_image_path))
                cv2.imwrite(extracted_image_path, extracted_image)

    if return_array:
        return extracted_images

def get_i(path,i_placeholder="$i", i_padding=5):
    try:
        max_path = max(glob.glob(path.replace(i_placeholder, "[0-9]"*i_padding)))
        search_pattern = "^{}$".format(path.replace(i_placeholder, "({})".format("[0-9]"*i_padding)))
        return int(re.findall(search_pattern, max_path)[0]) + 1
    except ValueError:
        return 0

if __name__ == "__main__":
    extract_detected(sys.argv[1], sys.argv[2], save_path=sys.argv[3])