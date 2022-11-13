#!/usr/bin/python3

from detection import *

def extract_detected(images, label, box_expand=[0.1, 0.1], model=None, save_path=None, i=None, i_placeholder="$i", i_padding=5, return_array=False, verbose=True):
    if type(images) is str:
        images = list(sorted(glob.glob(images)))
    elif type(images[0]) is str:
        images_globbed = []
        for image in images:
            if type(image) is str:
                images_globbed += list(glob.glob(images))
        
        images = images_globbed
            

    if model is None:
        model = load_model()

    if save_path and not i: 
        i = get_i(save_path)
    
    extracted_images = []

    for image in images:
        if type(image) == str:
            im_path = image
            image = cv2.imread(im_path)
        else:
            im_path = None
        
        results = detect_objects(image, model=model)
        for result in results:
            if result.label == label:
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
                    print("Extracted image from {} at {}".format(im_path, box))

                if return_array:
                    extracted_images.append(extracted_image)

                if save_path:
                    extracted_image_path = save_path.replace(i_placeholder, str(i).zfill(i_padding))
                    if verbose:
                        print("Saving image from {} at {} to {}".format(im_path, box, extracted_image_path))
                    cv2.imwrite(extracted_image_path, extracted_image)
                    i += 1

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