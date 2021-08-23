# Automated COVID Vaccination Card Verification

- [Automated COVID Vaccination Card Verification](#automated-covid-vaccination-card-verification)
  - [Tutorial](#tutorial)
  - [Requirements](#requirements)
  - [Installation & Setup](#installation--setup)
  - [Citations](#citations)

![terminal example](./assets/terminal_ex_duo_bigtext.png)

## Tutorial
- Part 1: [Automated COVID Vaccination Card Verification — Image Alignment](https://medium.com/reese-innovate/automated-covid-vaccination-card-verification-b27e289cf8b2)
- Part 2: [Automated COVID Vaccination Card Verification — Verification Checks](https://medium.com/reese-innovate/automated-covid-vaccination-card-verification-verification-checks-81ef451f59ef)

<!-- ## Usage

```
usage: verify_card.py [-h] -i IMAGE -t TEMPLATE [--tag TAG]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        path to input image
  -t TEMPLATE, --template TEMPLATE
                        path to the vaccination template image
  --tag TAG             prefix for output visualization files
```

- Terminal usage: `python verify_card.py --template templates/CDC_card_template_01.png --image <path-to-input>`
- We found there were slight variations (different aspect ratio, spacing between lines) in the CDC vaccination cards that were issued, so you can specify a specific template to verify against. We provide two templates (`./templates/CDC_card_template_01.png` and `./templates/CDC_card_template_02.png`). -->

## Requirements
- Python >= 3.7

## Installation & Setup
1. Install tesseract using [the official docs](https://tesseract-ocr.github.io/tessdoc/Installation.html).
    ```bash
    apt-get install tesseract-ocr
    ```
2. Clone and cd into this repo.
3. Install the package
    ```bash
    pip install .

    # or
    
    python setup.py install
    ```
4. Try out the example. Expected output:
    ```bash
    ❯ python example/verify_single_card.py
    [INFO] loading images...
    [INFO] aligning images...
    Valid vaccination card detected!
    ```
5. Try it out in your own script!
    ```python
    from vcv import verify_card

    ...

    isValid, failure_code = verify_card(img_path, template_path)
    ```

## Citations
- Adrian Rosebrock, Image alignment and registration with OpenCV, PyImageSearch, https://www.pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/, accessed on 6 May 2021
- Adrian Rosebrock, OCR a document, form, or invoice with Tesseract, OpenCV, and Python, PyImageSearch, https://www.pyimagesearch.com/2020/09/07/ocr-a-document-form-or-invoice-with-tesseract-opencv-and-python/, accessed on 6 May 2021
