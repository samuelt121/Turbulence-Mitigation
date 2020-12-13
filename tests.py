'''This script is used to test different modules of the code'''
import os
from utils import *
import numpy as np
import matplotlib.pyplot as plt  # plot graphs
from multiprocessing import Pool, Process, Queue
from skimage.transform import pyramid_gaussian, pyramid_laplacian
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1
from scipy import signal
from scipy import misc

import time


def f(x, y, q):
    a = x ** 2
    b = x ** 3 + y ** 2
    q.put([a, b])
# a, b = f(5,6)

def cart_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


if __name__ == '__main__':

# region create psf of circular aperture
    x = y = np.linspace(-10, 10, 1000)
    xx, yy = np.meshgrid(x, y)
    R = 5
    A = np.zeros(xx.shape)
    A[(xx-np.mean(x))**2 + (yy-np.mean(y))**2 < R**2] = 1

    # code not complete
    plt.imshow(A)
    A_fft = np.fft.fft2(A)
    A_fft = A_fft / max(A_fft)
    psf = abs(A_fft)**2
    plt.imshow(psf)

    # r, p = cart_to_polar(x, y)
    # u_eff = unit_r * np.pi / wavelength / fno
    # abs(2 * jinc(u_eff)) ** 2

    print('hello')
# endregion


# region Test spatial frequency
#     m_focal_length = 0.097
#     mm_pixel_pitch = 12
#
#     images_directory = os.path.dirname(os.path.abspath(__file__)) + "\\test_images\\"
#     ImagesSequence = loadImagesFromDir(images_directory,
#                                        [".png", ".bmp", ".jpg", ".JPG"])  # read all images in the relative
# # dir.
#     ImagesSequence = np.array(ImagesSequence).astype(np.uint8)
#     Image = ImagesSequence[2]
#     I2=cv2.resize(Image, (2000, 1125))
#     #roi = cv2.selectROI(I2)
#     roi = (3618, 912, 801, 399)
#
#     CropImage = Image[int(roi[1]):int(roi[1] + roi[3]),
#                                 int(roi[0]):int(roi[0] + roi[2])]
#
#     #showImage(CropImage)
#     #CropImage = Image
#     fft_of_ROI = np.fft.fftshift(np.fft.fft2(CropImage))
#     # fshx = np.fft.fftshift(np.fft.fftfreq((fft_of_ROI.shape[0])))
#     # fshy = np.fft.fftshift(np.fft.fftfreq((fft_of_ROI.shape[1])))
#     # fy = np.fft.fftfreq(fft_of_ROI.shape[0], d=2 * mm_pixel_pitch)
#     # fx = np.fft.fftfreq(fft_of_ROI.shape[1], d=2 * mm_pixel_pitch)
#     #
#     #
#     # fx_v, fy_v = np.meshgrid(fx, fy)
#     #angular_frequency = np.array(1000 * m_focal_length * np.sqrt(fx_v ** 2 + fy_v ** 2))
#
#     avg_delta = 8
#     vert_cross = np.abs(np.average(fft_of_ROI[:, int(np.floor(roi[3]/2) - avg_delta/2): int(np.floor(roi[3]/2) + avg_delta/2)], axis=1))
#     PSD_fft_ROI = fft_of_ROI * np.conj(fft_of_ROI) / (roi[2] * roi[3])
#     PSD_vert_cross = np.average(PSD_fft_ROI[:, int(np.floor(roi[3]/2) - avg_delta/2): int(np.floor(roi[3]/2) + avg_delta/2)], axis=1)
#
#     plt.figure(figsize=(64 * 4, 48 * 4), constrained_layout=False)
#     plt.subplot(141), plt.imshow(np.log(1 + np.abs(fft_of_ROI)), "gray"), plt.title("Centered Spectrum")
#     plt.subplot(142), plt.imshow(CropImage, "gray"), plt.title("Original Image")
#     plt.subplot(143), plt.plot(vert_cross), plt.title("Vertical cross of FFT")
#     plt.subplot(144), plt.plot(PSD_vert_cross), plt.title("Vertical cross of PSD")
#     plt.show()

    # endregion

# region Test 2dCorrelation
#     face = misc.face(gray=True) - misc.face(gray=True).mean()
#     template = np.copy(face[300:365, 670:750])  # right eye
#     #template -= template.mean()
#     face = face + np.random.randn(*face.shape) * 50  # add noise
#     corr = signal.correlate2d(face, template, boundary='symm', mode='same')
#     y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
#     # show images
#     fig, (ax_orig, ax_template, ax_corr) = plt.subplots(1, 3)
#     ax_orig.imshow(face, cmap='gray')
#     ax_orig.set_title('Original')
#     ax_orig.set_axis_off()
#     ax_template.imshow(template, cmap='gray')
#     ax_template.set_title('Template')
#     ax_template.set_axis_off()
#     ax_corr.imshow(corr, cmap='gray')
#     ax_corr.set_title('Cross-correlation')
#     ax_corr.set_axis_off()
#     ax_orig.plot(x, y, 'ro')
#     fig.show()
#     cv2.waitKey()
#     # endregion


# region Test Laplacian pyramid reconstruction
    ##Load images
    # dirPath = os.path.dirname(os.path.abspath(__file__))
    # images_directory = dirPath + "\\test_images\\"
    # ImagesSequence = loadImagesFromDir(images_directory, [".png",".bmp", ".jpg"]) # read all images in the relative dir.
    #
    # Image = ImagesSequence[0]
    # print(Image.shape)
    #
    # no_of_pyramid_levels = 4
    # GaussPyramid, LaplacianPyramid = createGaussianAndLaplacianPyramid(Image, no_of_pyramid_levels)
    # #GaussPyramid, LaplacianPyramid = createGradientLaplacianPyramid(Image, no_of_pyramid_levels)
    #
    # # LaplacianPyramid = pyramid_laplacian(Image, max_layer =no_of_pyramid_levels-1,  downscale=1.5)
    # recon_I = reconstructImageFromLaplacPyramid(LaplacianPyramid)
    # # recon_I = cv2.resize(recon_I, tuple(Image.shape)[::-1], interpolation=cv2.INTER_CUBIC)
    #
    # diff_I = abs(Image-recon_I)
    # difference = sum(sum(diff_I))
    # print('diff is', difference)
    # # GaussPyramid2, LaplacianPyramid2 = createGaussianAndLaplacianPyramid2(Image, no_of_pyramid_levels)
    # # recon_I2 = reconstructImageFromLaplacPyramid2(LaplacianPyramid2)
    # # recon_I2 = cv2.resize(recon_I2, tuple(Image.shape)[::-1], interpolation=cv2.INTER_CUBIC)
    #
    # fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(5, 10))
    #
    # ax0.imshow(Image, cmap='gray', vmin=0, vmax=255)
    # ax0.set_title("Original Image")
    # ax0.set_axis_off()
    #
    # ax1.imshow(recon_I, cmap='gray', vmin=0, vmax=255)
    # ax1.set_title("Reconstructed Image")
    # ax1.set_axis_off()
    #
    # ax2.imshow(diff_I, cmap='gray', vmin=0, vmax=255)
    # ax2.set_title("Difference")
    # ax2.set_axis_off()
    #
    # fig.tight_layout()
    # plt.show()
    # endregion

# region Test image registraion via optical flow

    # images_directory = os.path.dirname(os.path.abspath(__file__)) + "\\test_images\\"
    # ImagesSequence = loadImagesFromDir(images_directory,
    #                                    [".png", ".bmp", ".jpg"])  # read all images in the relative dir.
    # ImagesSequence = np.array(ImagesSequence).astype(np.uint8)
    # # roi = cv2.selectROI(ImagesSequence[0])
    # roi = (344, 226, 686, 817)
    # CroppedImagesSequence = ImagesSequence[:, int(roi[1]):int(roi[1] + roi[3]),
    #                             int(roi[0]):int(roi[0] + roi[2])]
    #
    # ImagesSequence = CroppedImagesSequence
    # no_rows, no_cols = ImagesSequence[0].shape
    #
    # u, v = optical_flow_tvl1(ImagesSequence[0], ImagesSequence[1])
    #
    # # # via Pool:
    # # pool = Pool()
    # # result = pool.starmap(optical_flow_tvl1, [(Frame, ImagesSequence) for Frame in ToBeRegisteredSequence])
    # # u, v = zip(*result)
    #
    # row_coords, col_coords = np.meshgrid(np.arange(no_rows), np.arange(no_cols),
    #                                      indexing='ij')
    # RegisteredImage = warp(ImagesSequence[1], np.array([row_coords + v, col_coords + u]), mode='nearest',
    #                        preserve_range=True).astype(np.uint8)
    #
    # ShowRegistration(ImagesSequence[0], ImagesSequence[1], RegisteredImage, np.uint8)
    # endregion

# region Test vectorization indices

# indices = createPatchIndices((255, 255), (5, 5), (256, 256))
# #beta(np.arange(10))
# pass

# # Test max function - depth direction
# A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# B = [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
# C = [[1, 3, 5], [7, 9, 11], [8, 0, 10]]
#
# print(np.max([A, B, C], axis=0))
# endregion

# region Test multiple outputs when using Pool object from multiprocessing

# pool = Pool()
# a = pool.starmap(f, [(x, 10) for x in range(2, 6)])
#
# t1, t2 = zip(*a)
#
# # Could also:
# list1 = [x[0] for x in a] # - create list
# tuple2 = (x[1] for x in a) # - create tuple
# endregion

# region Test multiple outputs when using Process object from multiprocessing
# t_start = time.time()
# q = Queue()
# processes = [Process(target=f, args=(10, i, q))
#              for i in range(2, 10)]
#
# for p in processes:
#     p.start()
#
# for p in processes:
#     p.join()
#
# result = [q.get() for p in processes]
# print(result)
# print('Process time:', t_start - time.time())
#
#
# # pool = Pool()
# # a = pool.starmap(f, [(x, 10) for x in range(2, 6)])
#
# t1, t2 = zip(*result)
#
# # Could also:
# list1 = [x[0] for x in result] # - create list
# tuple2 = (x[1] for x in result) # - create tuple
# endregion

# region Test image enhancement

# no_of_pyramid_levels = 4  # levels of pyramids in Image Fusion algorithm.
# # Load images
# dirPath = os.path.dirname(os.path.abspath(__file__))
# images_directory = dirPath + "\\test_images\\"
# ImagesSequence = loadImagesFromDir(images_directory, [".png", ".bmp", ".jpg"])  # read all images in the
#
# laplacianPyramids = [createGaussianAndLaplacianPyramid(ImagesSequence[i], no_of_pyramid_levels)[1] for
#                      i in range(len(ImagesSequence))]
#
# newPyramid = combinePyramids(laplacianPyramids)
# reconstructImage = reconstructImageFromLaplacPyramid(newPyramid)
# # recon_I2 = cv2.resize(reconstructImage, tuple(ImagesSequence[0].shape)[::-1],
# # interpolation=cv2.INTER_CUBIC)
#
# fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(5, 10))
#
# ax0.imshow(ImagesSequence[0], cmap='gray', vmin=0, vmax=255)
# ax0.set_title("Image 1")
# ax0.set_axis_off()
#
# ax1.imshow(ImagesSequence[1], cmap='gray', vmin=0, vmax=255)
# ax1.set_title("Image 2")
# ax1.set_axis_off()
#
# ax2.imshow(reconstructImage, cmap='gray', vmin=0, vmax=255)
# ax2.set_title("Reconstructed Image")
# ax2.set_axis_off()
#
# fig.tight_layout()
# plt.show()
# endregion
