from pathlib import Path

from vcv import verify_card

if __name__ == "__main__":

    img_path = str(Path("./example/data/IMG_1396.jpg").resolve())
    template_path = str(Path("./templates/CDC_card_template_01.png").resolve())
    # isValid = verify_card(img_path, template_path)
    isValid, failure_code = verify_card(img_path, template_path, show=True, output_dir="./example/output", verbose=True)

    if isValid:
        print("Valid vaccination card detected!")
    else:
        print("Invalid vaccination card detected!")
