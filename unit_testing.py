import unittest
from image_generator import generate_images
from sloganGenerator import get_taglines
from object_detector import detect_object
from optimal_coordinates_finder import get_optimal_coordinates
from PIL import Image

# Below is a series of unit tests verifying the functionality of individual module except
# the final integrator. The final integrator cannot be tested simply using a unit test
# because it can only be tested usually manual visual inference.
class TestModules(unittest.TestCase):

    # This tests whether the image generated is of the right dimension.
    def test_image_generator_1(self):
        generate_images(1)
        generated_image_path = 'generated_image/background0.png'
        img = Image.open(generated_image_path)
        self.assertEqual(img.size, (256, 256), "The size of the generated image should be 256 x 256.")

    # This tests whether the image generated contains an object akin to a coffee cup.
    def test_image_generator_2(self):
        generate_images(1)
        generated_image_path = 'generated_image/background0.png'
        idxs, _ = detect_object(generated_image_path)
        self.assetEqual(len(idxs) > 0, True, 'The generated image should contain an object akin to a coffee cup.')

    # This tests whether the slogan generator outputs a string.
    def test_slogan_generator_1(self):
        slogans = get_taglines()
        chosen_slogan = random.choice(slogans)
        self.assertIsInstance(chosen_slogan, str, 'The output slogan should be a string.')

    # This tests whether the slogan obtained is not empty.
    def test_slogan_generator_2(self):
        slogans = get_taglines()
        chosen_slogan = random.choice(slogans)
        self.assertTrue(len(chosen_slogan) > 0, True, 'The output slogan should contain at least one letter.')

    # This tests whether the slogan generator outputs a string with the first letter is capital.
    def test_slogan_generator_3(self):
        slogans = get_taglines()
        chosen_slogan = random.choice(slogans)
        self.assertTrue(chosen_slogan[0].isupper(), str, 'The first letter of the output slogan should be capital.')

    # This tests whether the slogan obtained contains the word coffee to make the slogan relevant to our context.
    def test_slogan_generator_4(self):
        slogans = get_taglines()
        chosen_slogan = random.choice(slogans)
        self.assertTrue('Coffee' or 'coffee' in chosen_slogan, True, 'The slogan should contain the word coffee.')

    # This tests whether the slogan obtained is to be of suitable length.
    def test_slogan_generator_5(self):
        slogans = get_taglines()
        chosen_slogan = random.choice(slogans)
        is_of_suitable_length =  0 < len(chosen_slogan) < 100
        self.assertTrue(is_of_suitable_length, True, 'The slogan obtained is to be of suitable length.')

    # This tests whether the object detector can detect coffee cup object.
    def test_object_detector_1(self):
        coffee_sample_img = 'coffee_img.png'
        idxs, boxes = detect_object(coffee_sample_img)
        self.assertTrue(len(idxs) > 0, True, 'The object detector should be able to detect the coffee cup object inside'
                                             ' the sample image')

    # This tests whether the object detector can properly locate a suitable bounding box of the coffee cup object in a
    # sample image. This particularly tests whether the origin (upper left coordinate) of the bounding box falls
    # within a suitable range of coordinates obtained manually through visual inference.
    def test_object_detector_2(self):
        coffee_sample_img = 'coffee_img.png'
        idxs, boxes = detect_object(coffee_sample_img)
        box_origin_x = boxes[0]
        box_origin_y = boxes[1]
        has_detected_right_box_origin = 40 < box_origin_x < 100 and 60 < box_origin_y < 100
        self.assertTrue(has_detected_right_box_origin, True, 'The bounding box origin of the sample image should'
                                                             ' fall within a suitable range of coordinates')

    # This tests whether the object detector can properly locate a suitable bounding box of the coffee cup object in a
    # sample image. This particularly tests whether the width and length of the bounding box falls
    # within a suitable range of lengths obtained manually through visual inference.
    def test_object_detector_3(self):
        coffee_sample_img = 'coffee_img.png'
        idxs, boxes = detect_object(coffee_sample_img)
        box_width = boxes[2]
        box_height = boxes[3]
        has_detected_right_box_origin = 100 < box_width < 200 and 100 < box_height < 150
        self.assertTrue(has_detected_right_box_origin, True, 'The bounding box size of the sample image should'
                                                             ' fall within a suitable range of lengths')

    # This tests whether the optimal x-coordinate is of type integer.
    def test_optimal_coordinates_finder_1(self):
        coffee_sample_img = 'coffee_img.png'
        idxs, boxes = detect_object(coffee_sample_img)
        final_x, _ = get_optimal_coordinates(idxs, boxes)
        self.assertIsInstance(final_x, int, 'The output x-coordinate should be an int.')

    # This tests whether the optimal y-coordinate is of type integer.
    def test_optimal_coordinates_finder_2(self):
        coffee_sample_img = 'coffee_img.png'
        idxs, boxes = detect_object(coffee_sample_img)
        _, final_y = get_optimal_coordinates(idxs, boxes)
        self.assertIsInstance(final_y, int, 'The output y-coordinate should be an int.')

    # This tests whether the optimal coordinates finder produce a right range of x- and y- coordinates.
    def test_optimal_coordinates_finder_3(self):
        coffee_sample_img = 'coffee_img.png'
        idxs, boxes = detect_object(coffee_sample_img)
        final_x, final_y = get_optimal_coordinates(idxs, boxes)
        is_of_right_range = 0 < final_x < 256 and 0 < final_y < 256
        self.assertTrue(is_of_right_range, True, 'The coordiantes produced is of right range.')

    # This tests whether the module can locate a suitable region of optimal coordinates in an sample coffee image.
    # This appropriate region is obtained manually.
    def test_optimal_coordinates_finder_4(self):
        coffee_sample_img = 'coffee_img.png'
        idxs, boxes = detect_object(coffee_sample_img)
        final_x, final_y = get_optimal_coordinates(idxs, boxes)
        is_in_right_region = 128 < final_x < 180 and 128 < final_y < 200
        self.assertTrue(is_in_right_region, True, 'The optimal coordinate located should be within the appropriate'
                                                  ' plausible region as obtained manually')


if __name__ == '__main__':
    unittest.main()