# https://stackoverflow.com/questions/16373425/add-text-on-image-using-pil
# This code implements the module for merging with or without the use of a template in the form of two functions.
from PIL import Image, ImageDraw, ImageFont

def merge_without_template(tagline, image_path, x, y, output_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    # font = ImageFont.truetype("FontType\artsender.ttf", 16)
    font = ImageFont.truetype('arial', 8)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((x, y), tagline, (102, 17, 0), font=font)
    img.save(output_path)
    return img


def merge_on_template(slogan, image_path, output_path):
    font_size = 8
    template_width = 256
    template_length = 320
    template = Image.new('RGB', (template_width, template_length))

    # First paste the generate image onto the image region of the template
    coffee_img = Image.open(image_path)
    template.paste(coffee_img, (0, 0))

    font = ImageFont.truetype('arial', font_size)
    # Dynamically adjust the size of the slogan_text until it fits around 65% of template's width
    img_fraction = 0.60
    while font.getsize(slogan)[0] < img_fraction * template.size[0]:
        font_size += 1
        font = ImageFont.truetype('arial', font_size)

    # Dynamically adjust the location of the text according to the length of slogan and
    # font size so that it is centered at the slogan region of the template
    x_write_location = template_width / 2 - font.getsize(slogan)[0] / 2
    y_write_location = 256 + ((template_length - 256) / 2 - font.getsize(slogan)[1] / 2)

    draw = ImageDraw.Draw(template)
    draw.text((x_write_location, y_write_location), slogan, (255, 255, 255), font=font)
    template.save(output_path)
    return template
