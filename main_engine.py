# This main method runs the main code of the engine for generating final marketing materials to the final outputs folder.
# It can either run with or without the use of a template depending on the input arguments when executing this on terminal.

from image_generator import generate_images
from sloganGenerator import get_taglines
from object_detector import detect_object
from optimal_coordinates_finder import get_optimal_coordinates
from integrator import merge_without_template, merge_on_template
import random
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True, help="1 for merging without template, 2 for merging on template")
    args = vars(ap.parse_args())
    # if int(args['mode']) == 1:
    #     results = np.zeros((50, 256, 256, 3))
    # elif int(args["mode"]) == 2:
    #     results = np.zeros((50, 320, 256, 3))

    # Generate and save a set of images from trained MSG-GAN
    no_of_images = 50
    generate_images(no_of_images)

    # Generate tag lines from the trained LSTM model
    tagLines = get_taglines()
    for i in range(50):
        chosen_tagline = random.sample(tagLines, 1)[0]
        background_idx = random.sample(range(no_of_images), 1)[0]
        background_image = '{}/background{}.png'.format('generated_image', background_idx)

        idxs = []
        boxes = []
        # Use the detector to detect if there is any coffee object in the background_image, if not do it again on the next image and so on until
        # the detector can detect it and return its location
        while len(idxs) < 1:
            ## Get the object locations within the generated image by using YOLO
            idxs, boxes = detect_object(background_image)
            if len(idxs) < 1:
                background_idx = (background_idx + 1) % no_of_images
                background_image = '{}/background{}.png'.format('generated_image', background_idx)

        ## Get the optimal coordinates from the obtained coffee location for inserting the generated tag line into the generated image
        final_x, final_y = get_optimal_coordinates(idxs, boxes)

        ##Integrate generated image with tagline and save the image in the corresponding path
        if int(args["mode"]) == 1:
            result = merge_without_template(chosen_tagline, background_image, final_x, final_y,
                                            'final_outputs/output{}.png'.format(i))
        elif int(args["mode"]) == 2:
            result = merge_on_template(chosen_tagline, background_image, 'final_outputs/output{}.png'.format(i))
        # result = np.array(result)
        # results[i, :, :, :] = result

    # tl.visualize.save_images(results, [10, 5],
    #                          'results_50/results_50_1.png')


if __name__ == '__main__':
    main()
