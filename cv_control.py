from pathlib import Path
import glob
import math

import numpy as np
import cv2
import click

@click.group()
def main():
    pass


@main.command()
def see():
    """Simple camera feed demo"""
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def _get_markers():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _get_board(markers=None):
    if not markers:
        markers = _get_markers()
    return cv2.aruco.CharucoBoard_create(5, 7, 1, 0.6, markers)


@main.command()
def get_markers():
    """Save marker images in ./images directory"""
    markers = _get_markers()

    board = _get_board(markers)
    board_image = board.draw((500, 700))

    outdir = Path('./images')
    outdir.mkdir(exist_ok=True)

    cv2.imwrite(str(outdir / 'board.png'), board_image)


    diamond = np.full((800, 800), 255, dtype=np.uint8)
    for x, y in (100, 100), (500, 100), (100, 500), (500, 500), (300, 300):
        diamond[x:x+200, y:y+200] = 0
    ids = 0, 1, 2, 3
    for i, (x, y) in zip(ids, [(1, 2), (0, 1), (2, 1), (1, 0)]):
        S = int(200 * 0.6)
        image = cv2.aruco.drawMarker(markers, i, S)
        diamond[
            200-S//2+x*200:200+S//2+x*200,
            200-S//2+y*200:200+S//2+y*200,
        ] = image
    cv2.putText(
        diamond, ' '.join(str(i) for i in ids),
        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA,
    )
    cv2.imwrite(str(outdir / 'diamond.png'), diamond)


@main.command()
def calibrate():
    """Calibrate camera using printed-out board.png, save to calibration.npz"""
    cap = cv2.VideoCapture(0)

    board = _get_board()

    all_corners = []
    all_ids = []

    camera_matrix = None

    while True:
        ret, frame = cap.read()

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            return
        if key & 0xFF == ord('y'):
            break

        corners, ids, rejected = cv2.aruco.detectMarkers(frame, board.dictionary)
        if ids is not None and len(ids) > 10:
            #corners = np.stack(c for [c] in corners)
            n, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, frame, board,
            )
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
            cv2.aruco.drawDetectedMarkers(frame, corners)

            if key & 0xFF == ord('c'):
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

            if all_ids:
                calibration, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                        charucoCorners=all_corners,
                        charucoIds=all_ids,
                        board=board,
                        imageSize=frame.shape[:2],
                        cameraMatrix=None,
                        distCoeffs=None,
                )

        if camera_matrix is not None:
            h,  w = frame.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)

        for size, color in (3, (0, 0, 0)), (1, (255, 255, 255)):
            cv2.putText(
                frame, f"[{len(all_ids)} frames] 'c' to use frame; 'y' to save; 'q' to ragequit.",
                (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, size, cv2.LINE_AA,
            )

        # Display the resulting frame
        cv2.imshow('frame', frame)

    if camera_matrix is not None:
        with Path('calibration.npz').open('wb') as f:
            np.savez(f, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    cap.release()
    cv2.destroyAllWindows()


@main.command()
def drive():
    try:
        f = Path('calibration.npz').open('rb')
    except FileNotFoundError:
        camera_matrix = dist_coeffs = None
    else:
        with f:
            npz = np.load(f)
            camera_matrix = npz['camera_matrix']
            dist_coeffs = npz['dist_coeffs']
    print(camera_matrix)

    cap = cv2.VideoCapture(0)
    board = _get_board()

    circ = np.float32([[0, 0, 0], [3,0,0], [0,3,0], [0,0,-3]]).reshape((-1, 1, 3))
    CIRC_N = 64
    circ = np.zeros((CIRC_N*2+7, 1, 3))
    for i in range(CIRC_N):
        circ[i*2][0][0] = math.cos(math.tau/CIRC_N*i) * 2 / 3
        circ[i*2][0][1] = math.sin(math.tau/CIRC_N*i) * 2 / 3
        circ[i*2+1][0][0] = math.cos(math.tau/CIRC_N*(i+0.5)) * 2 / 3
        circ[i*2+1][0][1] = math.sin(math.tau/CIRC_N*(i+0.5)) * 2 / 3
        circ[i*2+1][0][2] = 1/4
    for i in range(3):
        c = math.cos(math.tau/3*i)
        s = math.sin(math.tau/3*i)
        circ[CIRC_N*2+1+i*2][0][0] = -c - s
        circ[CIRC_N*2+1+i*2][0][1] = s - c
        circ[CIRC_N*2+2+i*2][0][0] = c - s
        circ[CIRC_N*2+2+i*2][0][1] = -s - c

    while True:
        ret, frame = cap.read()
        #frame = cv2.imread('scsh.jpg')

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('s'):
            cv2.imwrite('scsh.jpg', frame)

        corners, ids, rejected = cv2.aruco.detectMarkers(frame, board.dictionary)
        if ids is not None:
            diamond_corners, diamond_ids = cv2.aruco.detectCharucoDiamond(
                frame, corners, ids, 1/0.6);

            if diamond_ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners)
                #cv2.aruco.drawDetectedDiamonds(frame, diamond_corners, diamond_ids)

                if camera_matrix is not None:
                    rvec, tvec, points = cv2.aruco.estimatePoseSingleMarkers(
                        diamond_corners, 1, camera_matrix, dist_coeffs,
                    )
                    #cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, r[0], t[0], 1)

                    for r, t, ids in zip(rvec, tvec, diamond_ids):

                        pts, proj = cv2.projectPoints(circ, r, t, camera_matrix, dist_coeffs)
                        #print(pts)
                        #frame = cv2.circle(frame, (int(pts[0][0][0]), int(pts[0][0][1])), 20, (0, 200, 255), -1)

                        print(r, t, ids)
                        center = pts[-7][0]
                        to_x = pts[0][0]
                        diff = to_x - center
                        rot = -math.atan2(diff[1], diff[0])
                        print(rot)
                        if rot < -1:
                            rot = -1
                        if rot > 1:
                            rot = 1

                        for i, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                            a = pts[CIRC_N*2+1+i*2][0]
                            b = pts[CIRC_N*2+2+i*2][0]
                            frame = cv2.line(
                                frame,
                                tuple(int(c) for c in a),
                                tuple(int(c) for c in b),
                                (140, 100, 100), 5,
                            )
                            mid = (a + b) / 2
                            ot = (a*(rot+1) + b*(1-rot)) / 2
                            frame = cv2.line(
                                frame,
                                tuple(int(c) for c in mid),
                                tuple(int(c) for c in ot),
                                color, 5,
                            )

                        frame = cv2.polylines(frame, [pts[:CIRC_N*2].astype(int)], True, (255, 255, 255), 3)

                        cv2.putText(
                            frame, str(ids[0]),
                            tuple(int(c) for c in pts[-7][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                        )


        cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
