import numpy as np
import cv2


def project_3d_points_to_image(
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
):
    """
    Projects 3D points to the image coordinate system.

    Args:
        points_3d (numpy.ndarray): 3D coordinates of the points (N x 3)
        camera_matrix (numpy.ndarray): Camera intrinsic matrix (3 x 3)
        dist_coeffs (numpy.ndarray): Lens distortion coefficients (1 x 5 or 1 x 8)
        rvec (numpy.ndarray): Rotation vector (3 x 1)
        tvec (numpy.ndarray): Translation vector (3 x 1)

    Returns:
        numpy.ndarray: Projected 2D coordinates of the points (N x 2)
    """

    image_points, _ = cv2.projectPoints(
        points_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    return image_points.reshape(-1, 2)
