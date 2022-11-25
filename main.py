
import numpy as np
import os
import cv2

from image import Image


if __name__ == '__main__':
    # get list of image names
    im_list = [file for file in os.listdir('./imgs') if file.endswith('.jpg')]

    # for each image, instantiate an Image object to calculate the Homography that map points from plane to image
    images = [Image(os.path.join('./imgs', im_file), debug=True) for im_file in im_list]

    #construct V to solve for b by stacking the output of im.construct_v() (Equation.(17))
    V=images[0].construct_v()
    for i in range(1,len(images)):
        sub_V=images[i].construct_v()
        V=np.vstack((V,sub_V))
    print(V)

    #find b using the SVD trick
    P, Q, RH = np.linalg.svd(V, full_matrices=True)
    eigen_vectorsV=RH.T
    b=eigen_vectorsV[:, eigen_vectorsV.shape[1] - 1]
    print('norm(V @ b) =', np.linalg.norm(V @ b))  # check if the dot product between V and b is zero
    b11, b12, b22, b13, b23, b33 = b.tolist()
    print('b.shape: ', b.shape)
    print('b11: ', b11)
    print('b12: ', b12)
    print('b22: ', b22)
    print('b13: ', b13)
    print('b23: ', b23)
    print('b33: ', b33)

    #find components of intrinsic matrix from Equation.(12)
    v0=(b12*b13-b11*b23)/(b11*b22-(b12**2))
    lamda=b33-((b13**2)+v0*(b12*b13-b11*b23))/b11
    alpha=np.sqrt(lamda/b11)
    beta=np.sqrt(lamda*b11/(b11*b22-(b12**2)))
    c=-b12*(alpha**2)*beta/lamda
    u0=(c*v0/alpha)-(b13*(alpha**2)/lamda)
    """
    v0 = 0
    alpha = 0
    beta = 0
    c = 0
    u0 = 0
    """

    print('----\nCamera intrinsic parameters:')
    print('\talpha: ', alpha)
    print('\tbeta: ', beta)
    print('\tc: ', c)
    print('\tu0: ', u0)
    print('\tv0: ', v0)
    cam_intrinsic = np.array([
        [alpha, c, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    # get camera pose
    for im in images:
        R, t = im.find_extrinsic(cam_intrinsic)
        if not im.debug:
            print('R = \n', R)
            print('t = ', t)

    if images[0].debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()