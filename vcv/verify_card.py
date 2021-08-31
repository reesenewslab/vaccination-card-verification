# USAGE:
# python verify_card.py --template templates/CDC_card_template_01.png --image scans/valid/covid_01.JPEG

import os
import re
import pkg_resources
from pathlib import Path
from typing import Tuple

import cv2
import imutils
import numpy as np
import pytesseract
from .pyimagesearch.alignment import align_images


def read_title(aligned, template, debug=False, output_dir='') -> str:
    # manually determine bounding box coordinate of template title
    title_loc = (0, 0, 1487, 160)

    # get title ROI
    (x, y, w, h) = title_loc
    roi = aligned[y:y + h, x:x + w]

    # show ROI
    if debug:
        thickness = 10
        half_thickness = int(thickness / 2)
        template_highlight_title_roi = cv2.rectangle(template.copy(), (x + half_thickness, y + half_thickness, w, h), (232, 104, 30), thickness=thickness)
        aligned_highlight_title_roi = cv2.rectangle(aligned.copy(), (x + half_thickness, y + half_thickness, w, h), (232, 104, 30), thickness=thickness)
        stacked = np.hstack([template_highlight_title_roi, aligned_highlight_title_roi])
        cv2.imwrite(os.path.join(output_dir, 'bb_title.jpg'), stacked)
        cv2.imwrite(os.path.join(output_dir, 'field_title.jpg'), roi)

    # OCR the ROI using Tesseract
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb)

    return text


def logo_template_match(aligned, template_img, debug=True, output_dir=''):
    """
    Returns the top-left 2D coordinate of the template match.
    """

    # first convert to grayscale
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    logo_template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    logo_template_h = logo_template_gray.shape[0]
    logo_template_w = logo_template_gray.shape[1]

    # perform Zero-normalized cross-correlation template match
    tm_result = cv2.matchTemplate(aligned_gray, logo_template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_zncc_val, _, max_zncc_loc = cv2.minMaxLoc(tm_result)  # highest value is the best match
    # print(f'max_zncc_val: {max_zncc_val}, max_zncc_loc: {max_zncc_loc}')

    if debug:
        # get the ROI coordinates for the match
        roi_x0 = min(aligned_gray.shape[1], max(0, max_zncc_loc[0]))
        roi_y0 = min(aligned_gray.shape[0], max(0, max_zncc_loc[1]))
        roi_x1 = min(aligned_gray.shape[1], max(0, max_zncc_loc[0] + logo_template_w))
        roi_y1 = min(aligned_gray.shape[0], max(0, max_zncc_loc[1] + logo_template_h))

        # visualize the match next to the template
        aligned_roi = aligned_gray[roi_y0:roi_y1, roi_x0:roi_x1]
        tm_stacked = np.hstack([aligned_roi, logo_template_gray])
        cv2.imwrite(os.path.join(output_dir, 'template_match.jpg'), tm_stacked)

    return max_zncc_val, max_zncc_loc


def perform_verification_checks(RANSAC_inliers, title, max_zncc_val, max_zncc_loc, verbose) -> Tuple[bool, int]:
    '''
    Perform 3 verification checks (ransac inlier, title verificaiton, and logo template match)

    Input:
        RANSAC_inliers
        title
        max_zncc_val
        max_zncc_loc

    Output: (bool, int)
        [bool] True if all checks passed. False otherwise.
        [int] failure code
            0: all checks passed
            1: failed RANSAC inlier verificaiton
            2: failed title verification
            3: failed CDC logo check -- location not in top right corner
            4: failed CDC logo check -- similarity score too low
    '''
    # require a minimum number of inliers during the homography estimation
    if RANSAC_inliers < 11:
        if verbose:
            print("Failed RANSAC inlier verification!")
        return (False, 1)

    # verify title with regex (allowing for variable amounts of whitespace)
    expectedTitleRegex = re.compile(r'(COVID\s*-\s*1\s*9\s*Vaccination\s*Record\s*Card)', re.DOTALL)
    match = expectedTitleRegex.search(title)
    if match is None:
        if verbose:
            print('Failed title verification!',
                '\nExpected title: COVID-19 Vaccination Record Card',
                f'\nDetected title: {title}')
        return (False, 2)

    # verify the CDC logo is near the top right corner
    if max_zncc_loc[0] < 1470 or max_zncc_loc[1] > 10:
        if verbose:
            print("Failed CDC logo check. Location should be in top right corner!")
        return (False, 3)
    elif max_zncc_val < 0.4:
        if verbose:
            print("Failed CDC logo check. Similarity score too low!")
        return (False, 4)

    return (True, 0)


def visualize_aligned(aligned, template, output_dir=''):
    # resize both the aligned and template images so we can easily
    # visualize them on our screen
    aligned = imutils.resize(aligned, width=700)
    template = imutils.resize(template, width=700)

    # our first output visualization of the image alignment will be a
    # side-by-side comparison of the output aligned image and the
    # template
    stacked = np.hstack([aligned, template])

    # our second image alignment visualization will be *overlaying* the
    # aligned image on the template, that way we can obtain an idea of
    # how good our image alignment is
    overlay = template.copy()
    output = aligned.copy()
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

    cv2.imwrite(os.path.join(output_dir, 'stacked.jpg'), stacked)
    cv2.imwrite(os.path.join(output_dir, 'overlay.jpg'), output)


def verify_card(image_path: str, template_path: str=None, show: bool=False, output_dir: str='./output', verbose: bool=False) -> Tuple[bool, int]:
    '''
    Verifys a given vaccination card matches a template.

    Input:
        image_path: A string specifying the path to the input image.
        [Optional] template_path: Specify your own vaccination card template. Defaults to template included in the package.
        [Optional] show: Writes intermediate images if True.
        [Optional] output_dir: A string specifying a directory where output images should be written. Only relevant if `show` is True.
        [Optional] verbose: Verbosity mode. Prints failure reason to stdout.

    Output: (bool, int)
        [bool] True if all checks passed. False otherwise. 
        [int] failure code.
            0: all checks passed
            1: failed RANSAC inlier verificaiton
            2: failed title verification
            3: failed CDC logo check -- location not in top right corner
            4: failed CDC logo check -- similarity score too low
    '''
    # set default resource paths if not specified
    if template_path is None:
        template_path = pkg_resources.resource_filename(__name__, "templates/CDC_card_template_01.png")
    logo_file_path = pkg_resources.resource_filename(__name__, "templates/cdc_logo_template.png")

    # check the input paths exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image can not be found. (image_path='{image_path}')")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template image can not be found. (template_path='{template_path}')")
    if not os.path.exists(logo_file_path):
        raise FileNotFoundError(f"Logo template image can not be found. (logo_file_path='{logo_file_path}')")

    # load the input image and templates from disk
    if verbose:
        print("[INFO] loading images...")
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)
    logo_template = cv2.imread(logo_file_path)

    # check if images loaded correctly
    if image is None:
        raise ValueError("Input image is empty, or unable to load.")
    if template is None:
        raise ValueError("Input template is empty, or unable to load.")

    # create the output directory
    if show:
        output_dir = os.path.join(output_dir, Path(image_path).stem)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # align the images
    if verbose:
        print("[INFO] aligning images...")
    aligned, RANSAC_inliers = align_images(image, template, debug=show, output_dir=output_dir)

    # visualize aligned images
    if show:
        visualize_aligned(aligned, template, output_dir=output_dir)

    # perform OCR on the aligned image
    title = read_title(aligned, template, debug=show, output_dir=output_dir)

    # perform logo template match
    max_zncc_val, max_zncc_loc = logo_template_match(aligned, logo_template, debug=show, output_dir=output_dir)

    # verify it is a valid vaccination card
    verified, failure_code = perform_verification_checks(RANSAC_inliers, title, max_zncc_val, max_zncc_loc, verbose)
    
    return (verified, failure_code)


