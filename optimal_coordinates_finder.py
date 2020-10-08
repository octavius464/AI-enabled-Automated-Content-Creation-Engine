# This function obtains the optimal coordinate from the image for merging the slogan text onto an image. It achieves
# this by gathering the bounding boxes information for objects detected by the object detector module, and then it filters
# out all the possible coordinates which does not overlap with the object or overlow, and locate an optimal position
# using the Manhatten heuristics.
def get_optimal_coordinates(idxs, boxes):
    tagline_width = 35  # for 360x360 image, use 100
    tagline_height = 15  # for 360x360 image, use 50
    image_size = 256
    possible_coordinates = {}
    for y in range(1, image_size):
        for x in range(1, image_size):
            possible_coordinates[(x, y)] = 0
    for b in idxs.flatten():
        # extract the bounding box coordinates (top left coordinate)
        (x_box, y_box) = (boxes[b][0], boxes[b][1])
        (w_box, h_box) = (boxes[b][2], boxes[b][3])

        # Erase those coordinates which can potentially overlap with the bounding box
        for y in range(1, image_size):
            for x in range(1, image_size):
                if x > x_box - tagline_width and x < x_box + w_box and y > y_box - tagline_height and y < y_box + h_box:
                    if (x, y) in possible_coordinates:
                        del possible_coordinates[(x, y)]

    for (x, y), _ in possible_coordinates.items():
        possible_coordinates[(x, y)] = abs(x - image_size / 2) + abs(y - image_size / 2)  # for 360x360 image, use 180

    # For the remaining coordinates, favour those which is more centrally-located by calculating the Manhatten heuristics scores.
    minScore = 100000
    final_x = 0
    final_y = 0
    for (x, y), _ in possible_coordinates.items():
        if possible_coordinates[(x, y)] < minScore:
            minScore = possible_coordinates[(x, y)]
            final_x = x
            final_y = y
    return final_x, final_y
