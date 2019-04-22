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


class Object2D:
    def __init__(self, corners, *, angle=None):
        self.center = corners.sum(0) / 4
        y1 = corners[:2].sum(0) / 2
        if angle is None:
            self.angle = math.atan2(y1[1]-self.center[1], y1[0]-self.center[0]) + math.tau/4
        else:
            self.angle = angle
        self.size = np.linalg.norm(self.center - y1) * 2

        c, s = np.cos(-self.angle), np.sin(-self.angle)
        self.rotation_matrix = np.array(((c, -s), (s, c)))

    def draw(self, frame, color=(0, 255, 0)):
        arrow = np.array([[-0.5, 1], [0.5, 1], [0, -1]]) * 10
        arrow = arrow @ self.rotation_matrix
        arrow += self.center
        cv2.polylines(frame, [arrow.astype(int)], True, color, 1)


@main.command()
def drive():
    try:
        f = Path('calibration.npz').open('rb')
    except FileNotFoundError:
        camera_matrix = dist_coeffs = None
        drawing = False
    else:
        with f:
            npz = np.load(f)
            camera_matrix = npz['camera_matrix']
            dist_coeffs = npz['dist_coeffs']
        drawing = True

    cap = cv2.VideoCapture(0)
    board = _get_board()

    diagram_points = np.zeros((7, 1, 3))
    for i in range(3):
        c = math.cos(math.tau/3*(-i-1))
        s = math.sin(math.tau/3*(-i-1))
        diagram_points[i*2][0][0] = -c - s
        diagram_points[i*2][0][1] = s - c
        diagram_points[i*2+1][0][0] = c - s
        diagram_points[i*2+1][0][1] = -s - c

    target = None
    def set_target(event,x,y,flags,param):
        nonlocal target
        if event == cv2.EVENT_LBUTTONDOWN:
            array = np.array([[x-10, y-10], [x+10, y+10], [x-10, y+10], [x+10, y+10]])
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                target = Object2D(array, angle=90)
            else:
                target = Object2D(array)
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', set_target)

    while True:
        ret, frame = cap.read()
        #frame = cv2.imread('scsh.jpg')
        frame = cv2.imread('scsh.png')
        h,  w = frame.shape[:2]

        if drawing:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('s'):
            cv2.imwrite('scsh.jpg', frame)

        corners, ids, rejected = cv2.aruco.detectMarkers(frame, board.dictionary)

        if target is None:
            target = Object2D(np.array([[0, 0], [w, 0], [0, h], [w, h]]))
        if drawing:
            target.draw(frame)

        if ids is not None:
            diamond_corners, diamond_ids = cv2.aruco.detectCharucoDiamond(
                frame, corners, ids, 1/0.6)
            f = 255
            for corner, color in zip(diamond_corners, ((0,f,0), (f,0,0), (0,0,f), (f,f,0), (0,f,f), (f,0,f))):
                if diamond_ids is not None:
                    car = Object2D(corner.reshape((4, 2)))
                    if drawing:
                        car.draw(frame, color)

                    ANGLE_SATURATION = 0.7  # rad
                    DISTANCE_SATURATION = 4  # charuco tiles
                    DISTANCE_PRIORITY = 2
                    MOTOR_MAX = 1023

                    angle_diff = target.angle - car.angle
                    while angle_diff > math.tau/2:
                        angle_diff -= math.tau
                    while angle_diff < -math.tau/2:
                        angle_diff += math.tau
                    if angle_diff > ANGLE_SATURATION:
                        angle_diff = ANGLE_SATURATION
                    if angle_diff < -ANGLE_SATURATION:
                        angle_diff = -ANGLE_SATURATION

                    drive_vector = target.center - car.center
                    drive_vector /= car.size * DISTANCE_SATURATION
                    drive_vector = car.rotation_matrix @ drive_vector
                    drive_vector *= 1, -1
                    norm = np.linalg.norm(drive_vector)
                    if norm > 1:
                        drive_vector /= norm
                        norm = 1

                    if drawing:
                        frame = cv2.line(
                            frame,
                            tuple(int(c) for c in drive_vector*[1,-1]*50 + [w//2, h//2]),
                            (w//2, h//2),
                            color, 1,
                        )

                    speeds = np.zeros(3)
                    for i in range(3):
                        angle = math.atan2(drive_vector[1], drive_vector[0]) + math.tau/6
                        motor_angle = math.tau / 3 * i
                        speeds[i] = int(
                            math.cos(motor_angle - angle) * norm * MOTOR_MAX * DISTANCE_PRIORITY
                            + angle_diff * MOTOR_MAX / ANGLE_SATURATION
                        )
                    mx = max(speeds)
                    if mx > MOTOR_MAX:
                        speeds = speeds / mx * MOTOR_MAX
                    mn = min(speeds)
                    if mn < -MOTOR_MAX:
                        speeds = speeds / mn * -MOTOR_MAX

                    if drawing:
                        rvec, tvec, points = cv2.aruco.estimatePoseSingleMarkers(
                            [corner], 1, camera_matrix, dist_coeffs,
                        )

                        for r, t, ids in zip(rvec, tvec, diamond_ids):

                            pts, proj = cv2.projectPoints(
                                np.array([[[drive_vector[0], drive_vector[1], 0]], [[0, 0, 0]]]),
                                r, t, camera_matrix, dist_coeffs,
                            )
                            frame = cv2.polylines(frame, [pts.astype(int)], True, (140, 100, 100), 5)

                            pts, proj = cv2.projectPoints(diagram_points, r, t, camera_matrix, dist_coeffs)

                            for i, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                                a = pts[i*2][0]
                                b = pts[i*2+1][0]
                                frame = cv2.line(
                                    frame,
                                    tuple(int(c) for c in a),
                                    tuple(int(c) for c in b),
                                    (140, 100, 100), 5,
                                )
                                rot = speeds[i] / 1000
                                mid = (a + b) / 2
                                frame = cv2.line(
                                    frame,
                                    tuple(int(c) for c in mid),
                                    tuple(int(c) for c in mid),
                                    (140, 100, 100), 20,
                                )
                                ot = (a*(rot+1) + b*(1-rot)) / 2
                                frame = cv2.line(
                                    frame,
                                    tuple(int(c) for c in mid),
                                    tuple(int(c) for c in ot),
                                    color, 5,
                                )

                                for color, size in ((0, 0, 0), 2), ((255, 255, 255), 1):
                                    cv2.putText(
                                        frame, str(int(speeds[i])),
                                        tuple(int(c) for c in mid),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, size, cv2.LINE_AA,
                                    )

                            cv2.putText(
                                frame, str(ids[0]),
                                tuple(int(c) for c in pts[-1][0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                            )


        if drawing:
            cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
