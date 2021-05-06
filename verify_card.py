# USAGE:
# python verify_card.py --template templates/CDC_card_template_01.png --image scans/valid/covid_01.JPEG

import argparse
import re

import cv2
import imutils
import numpy as np
import pytesseract
from pyimagesearch.alignment import align_images


def read_title(aligned, template, fileTag=None, debug=False) -> str:
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
        cv2.imwrite(f"output/{fileTag}_bb_title.jpg", stacked)
        cv2.imwrite(f"output/{fileTag}_field_title.jpg", roi)

    # OCR the ROI using Tesseract
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb)

    return text


def logo_template_match(aligned, template_img, fileTag=None, debug=True):
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
        cv2.imwrite(f"output/{fileTag}_TM.jpg", tm_stacked)

    return max_zncc_val, max_zncc_loc


def verify_card(RANSAC_inliers, title, max_zncc_val, max_zncc_loc) -> bool:
    # require a minimum number of inliers during the homography estimation
    if RANSAC_inliers < 11:
        print("Failed RANSAC inlier verification!")
        return False

    # verify title with regex (allowing for variable amounts of whitespace)
    expectedTitleRegex = re.compile(r'(COVID\s*-\s*1\s*9\s*Vaccination\s*Record\s*Card)', re.DOTALL)
    match = expectedTitleRegex.search(title)
    if match is None:
        print('Failed title verification!',
              '\nExpected title: COVID-19 Vaccination Record Card',
              f'\nDetected title: {title}')
        return False

    # verify the CDC logo is near the top right corner
    if max_zncc_loc[0] < 1470 or max_zncc_loc[1] > 10:
        print("Failed CDC logo check. Location should be in top right corner!")
        return False
    elif max_zncc_val < 0.4:
        print("Failed CDC logo check. Similarity score too low!")
        return False

    return True


def visualize_aligned(aligned, template, fileTag=None):
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

    # show the two output image alignment visualizations
    cv2.imwrite(f"output/{fileTag}_stacked.jpg", stacked)
    cv2.imwrite(f"output/{fileTag}_overlay.jpg", output)


if __name__ == "__main__":

    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image that we'll align to template")
    ap.add_argument("-t", "--template", required=True,
                    help="path to input template image")
    ap.add_argument("--tag", required=False,
                    help="prefix for output files")
    args = vars(ap.parse_args())

    # load the input image and template from disk
    print("[INFO] loading images...")
    image = cv2.imread(args["image"])
    template = cv2.imread(args["template"])
    logo_template = cv2.imread('templates/cdc_logo_template.png')

    # align the images
    print("[INFO] aligning images...")
    aligned, RANSAC_inliers = align_images(image, template, args["tag"], debug=True)

    # visualize aligned images
    visualize_aligned(aligned, template, fileTag=args["tag"])

    # perform OCR on the aligned image
    title = read_title(aligned, template, fileTag=args["tag"], debug=True)

    # perform logo template match
    max_zncc_val, max_zncc_loc = logo_template_match(aligned, logo_template, fileTag=args["tag"], debug=True)

    # verify it is a valid vaccination card
    verified = verify_card(RANSAC_inliers, title, max_zncc_val, max_zncc_loc)
    if verified:
        print("Valid vaccination card detected!")
    else:
        print("Invalid vaccination card detected!")
