import numpy as np
import cv2

def pnp(points_3d, points_2d, camera_matrix,method=cv2.SOLVEPNP_ITERATIVE):
    try:
        dist_coeffs = pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method==cv2.SOLVEPNP_EPNP:
        points_3d=np.expand_dims(points_3d, 0)
        points_2d=np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    _, R_exp, t = cv2.solvePnP(points_3d,
                               points_2d,
                               camera_matrix,
                               dist_coeffs,
                               flags=method)
                              # , None, None, False, cv2.SOLVEPNP_UPNP)

    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)

    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return np.concatenate([R, t], axis=-1)

def ransacpnp(points_3d, points_2d, camera_matrix,method=cv2.SOLVEPNP_ITERATIVE):
    try:
        dist_coeffs = pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method==cv2.SOLVEPNP_EPNP:
        points_3d=np.expand_dims(points_3d, 0)
        points_2d=np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)

    success, R_exp, t, inliers = cv2.solvePnPRansac(points_3d,
                               points_2d,
                               camera_matrix,
                               dist_coeffs,
                               reprojectionError=12.0,
                               flags=method)

    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return np.concatenate([R, t], axis=-1)