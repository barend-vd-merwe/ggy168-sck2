import streamlit as st
import cv2 as cv
import numpy as np
import gc

def submission_viewer(img_url, height=1000, max_zoom_ratio=4.0, key="osd_viewer"):
    # openseadragon
    html_osd = f"""
    <!DOCTYPE html>
    <html style="height:100%">
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>
        <style>
            body {{
    margin: 0;
    padding: 0;
    background-color: #F5F5DC;
    height: 100%;
    width: 100%;
    overflow: hidden;}}
            #openseadragon-container {{
                width: 100vw;
                height: 100vh;
                background: #F5F5DC;
                border: 10px solid #374151;
                border-radius: 8px;
    box-sizing: border-box;
            }}
        </style>
    </head>
    <body>
        <div id="openseadragon-container"></div>
        <script type="text/javascript">
            var viewer = OpenSeadragon({{
                id: "openseadragon-container",
                prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
                tileSources: {{
                    type: 'image',
                    url:  '{img_url}'
                }},
                // Performance and Zoom configuration
                maxZoomPixelRatio: {max_zoom_ratio},
                minZoomImageRatio: 0.8,
                visibilityRatio: 0.8,
                constrainDuringPan: true,
                
                // Controls setup
                showNavigationControl: true,
                navigationControlAnchor: OpenSeadragon.ControlAnchor.TOP_LEFT,
                showRotationControl: false,
                showFullPageControl: true,
                showHomeControl: true,
                
                // Smooth interaction tuning
                animationTime: 0.3,
                bllendTime: 0.1,
                springStiffness: 10
            }});
        </script>
    </body>
    </html>
    """
    st.iframe(html_osd, height=height)

def aruco_align(image, ac):
    method = "Aruco Alignment"
    aruco_dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 100
    parameters.cornerRefinementMinAccuracy = 0.01

    detector = cv.aruco.ArucoDetector(aruco_dictionary, parameters)

    # use detector to detect aruco markers
    corners, ids, _ = detector.detectMarkers(image)

    # get the coordinates of the center points of each aruco marker
    ids = ids.flatten().tolist()

    aruco_0_index = ids.index(0)
    aruco_0_corners = corners[aruco_0_index]
    aruco_0_corners_rshp = aruco_0_corners.reshape((4,2))

    aruco_1_index = ids.index(1)
    aruco_1_corners = corners[aruco_1_index]
    aruco_1_corners_rshp = aruco_1_corners.reshape((4,2))

    aruco_2_index = ids.index(2)
    aruco_2_corners = corners[aruco_2_index]
    aruco_2_corners_rshp = aruco_2_corners.reshape((4,2))

    aruco_3_index = ids.index(3)
    aruco_3_corners = corners[aruco_3_index]
    aruco_3_corners_rshp = aruco_3_corners.reshape((4,2))

    aruco_markers = [aruco_0_corners_rshp,
                     aruco_1_corners_rshp,
                     aruco_2_corners_rshp,
                     aruco_3_corners_rshp]

    current_coords_list = []
    for marker in aruco_markers:
        for i in range(4):
            x_coord = marker[i,0]
            y_coord = marker[i,1]
            current_coords_list.append([x_coord, y_coord])
    current_coords = np.array(current_coords_list, dtype=np.float32)

    # Set target coordinates

    target_coords_list = []

    for i in range(len(ac)):
        x_coord = ac.iloc[i,4]
        y_coord = ac.iloc[i,5]
        target_coords_list.append([x_coord, y_coord])

    target_coords = np.array(target_coords_list, dtype=np.float32)

    # transform the image
    homography_matrix, mask = cv.findHomography(current_coords, target_coords, cv.RANSAC, 2.5)
    warped_img = cv.warpPerspective(image,
                                    homography_matrix,
                                    (2480, 3508),
                                    flags=cv.INTER_LINEAR,
                                    borderMode=cv.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))

    del image
    gc.collect()

    return warped_img, method

def sift_align(template, target):
    method = "SIFT Alignment"
    gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    gray_target = cv.cvtColor(target, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    keypoints_template, descriptors_template = sift.detectAndCompute(gray_template, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(gray_target, None)

    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck = True)
    matches = matcher.match(descriptors_template, descriptors_target)

    # good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    # if len(good_matches) < 4:
    #     raise ValueError("Not enough matching points found between the images.")
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 200.0]
    if len(good_matches) < 4:
        raise ValueError("Not enough cross-checked matching points found between the images.")

    target_pts = np.float32([keypoints_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    template_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    h_matrix, _ = cv.findHomography(target_pts, template_pts, cv.RANSAC, 2.5)
    
    warped_img =  cv.warpPerspective(target,
                                     h_matrix,
                                     (2480,3508),
                                     borderMode=cv.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))

    return warped_img, method


def dms_to_dd(dms):
    raw = dms.strip()
    parts = raw.split("/")
    nums = [float(p) for p in parts]
    if len(nums) != 3:
        st.error("Incorrect DMS entry. Entry must be dd/mm/ss (if there are no seconds, just enter 00).")
    else:
        try:
            deg, minutes, sec = [float(p) for p in parts]
            md = round(deg + (minutes / 60.0) + (sec / 3600.0), 4)
        except:
            st.error("Incorrect data entered.")
    return md
