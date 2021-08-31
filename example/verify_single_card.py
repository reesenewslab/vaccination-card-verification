import os

from vcv import verify_card

if __name__ == "__main__":

    example_dir = os.path.dirname(__file__)
    img_path = os.path.join(example_dir, 'data', 'IMG_1396.jpg')
    output_dir = os.path.join(example_dir, 'output')
    # isValid = verify_card(img_path)
    isValid, failure_code = verify_card(img_path, show=True, output_dir=output_dir, verbose=True)

    if isValid:
        print("Valid vaccination card detected!")
    else:
        print("Invalid vaccination card detected!")
