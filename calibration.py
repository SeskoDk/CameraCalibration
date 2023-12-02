import cv2
import datetime
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt


class ChessboardCapture:
    def __init__(self, image_dir: str = ".\calib_images") -> None:
        self.dir = Path(image_dir)
        if not self.dir.exists():
            self.dir.mkdir()

        self.camera = cv2.VideoCapture(0)

    def capture(self) -> None:
        """
        This method captures images from a camera and saves them using a timestamp as the filename.
        """
        while True:
            ret, frame = self.camera.read()
            cv2.imshow("Press [space] to take a photo or [q] to quit", frame)
            key = cv2.waitKey(1)
            timestemp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

            if key == ord(" "):
                filename = f"{timestemp}.jpg"
                file = dir / filename
                cv2.imwrite(str(file), frame)
            elif key == ord("q"):
                break

        self.camera.release()
        cv2.destroyAllWindows()


class ChessboardCornerCapture(ChessboardCapture):
    def __init__(
        self,
        image_dir: str = ".\calib_images",
        n_Corners_W: int = 8,
        n_Corners_H: int = 5,
    ) -> None:
        super().__init__(image_dir)
        self.nC_W = n_Corners_W
        self.nC_H = n_Corners_H
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def recordInnerCorners(self, outputfile="recording", video_fps: int = 30):
        """
        Records a video of a chessboard, detects inner corners, and saves it as an mp4 file.
        """
        video_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_file = f"{outputfile}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_file, fourcc, video_fps, (video_width, video_height)
        )

        while True:
            ret, frame = self.camera.read()
            key = cv2.waitKey(1)

            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(
                    gray, (self.nC_W, self.nC_H), None
                )

                if ret:
                    corners2 = cv2.cornerSubPix(
                        gray, corners.squeeze(), (11, 11), (-1, -1), self.criteria
                    )
                    cv2.drawChessboardCorners(
                        frame, (self.nC_W, self.nC_H), corners2, ret
                    )

                cv2.imshow("Chessboard Corners", frame)
                out.write(frame)

            if key == ord("q"):
                break

        self.camera.release()
        out.release()
        cv2.destroyAllWindows()


class CameraCalibration:
    def __init__(
        self,
        image_dir: str = r".\calib_images",
        undist_path: str = r".\undist_images",
        n_Corners_W: int = 8,
        n_Corners_H: int = 5,
        square_dist_in_m: float = 0.31,
    ) -> None:
        self.files = [str(file) for file in Path(image_dir).rglob("*.jpg")]
        self.undist_path = Path(undist_path)
        self.CBCapture = ChessboardCornerCapture(image_dir, n_Corners_W, n_Corners_H)
        self.criteria = self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        self.nC_W = n_Corners_W
        self.nC_H = n_Corners_H

        objp = np.zeros((n_Corners_W * n_Corners_H, 3), np.float32)
        objp[:, :2] = np.mgrid[0:n_Corners_W, 0:n_Corners_H].T.reshape(-1, 2)
        self.objp = objp * square_dist_in_m
        self.image_corners = list()
        self.objp_per_images = list()

        self.K = None
        self.dist_coeffs = None
        self.extrinsic_matrix_per_image = list()

    def get_intrinsic(self):
        return self.K

    def get_dist_coeffs(self):
        return self.dist_coeffs

    def get_extrinsics(self):
        return self.extrinsic_matrix_per_image

    def calc_intrinsic_and_distortion_coefficients(self):
        for file in self.files:
            image = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, (self.nC_W, self.nC_H), None)
            if ret == True:
                corners = cv2.cornerSubPix(
                    gray, corners.squeeze(), (11, 11), (-1, -1), self.criteria
                )
                self.image_corners.append(corners)

        cv2.destroyAllWindows()

        self.objp_per_images  = [self.objp for _ in range(len(self.image_corners))]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objp_per_images , self.image_corners, gray.shape[::-1], None, None
        )
        rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
        extrinsic_matrix = np.hstack((rotation_matrix, tvecs[0]))

        self.K = mtx
        self.dist_coeffs = dist
        self.extrinsic_matrix_per_image.append(extrinsic_matrix)

    def undistord_images(self):
        K = self.get_intrinsic()
        dist_coeffs = self.get_dist_coeffs()

        if self.undist_path.exists():
            shutil.rmtree(self.undist_path)
        self.undist_path.mkdir()

        for idx, file in enumerate(self.files):
            image = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            h, w = image.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                K, dist_coeffs, (w, h), 1, (w, h)
            )
            undist_image = cv2.undistort(image, K, dist_coeffs, None, newcameramtx)

            timestemp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            filename = f"{timestemp}_{idx}.jpg"
            file_path = self.undist_path / filename
            cv2.imwrite(str(file_path), undist_image)

        cv2.destroyAllWindows()


def main():
    CC = CameraCalibration()
    CC.calc_intrinsic_and_distortion_coefficients()
    CC.undistord_images()


if __name__ == "__main__":
    main()
