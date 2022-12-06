import cv2
import h5py
import os
import os.path as osp
from skimage import io
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric
from dipy.viz import regtools


def get_matching_score(pws_mask, dv_mask):
    """
    This function calculates the similarity between the shapes of two images.
    The smaller the value returned, the higher the similarity between the two images
    :param pws_mask: the PWS img
    :param dv_mask: the confocal img
    :return: return the matching score, the smaller, the better
    """
    threshPWS = np.uint8(pws_mask)
    threshDV = np.uint8(dv_mask)

    def find_longest_cnt(contours: tuple):
        cnt_len = [cnt.shape[0] for cnt in contours]
        return np.argmax(cnt_len)

    # extract all possible contours from the image
    contour_point, hierarchy_PWS = cv2.findContours(threshPWS, 2, 1)
    cnt_PWS = contour_point[find_longest_cnt(contour_point)]
    contours_DV, hierarchy_DV = cv2.findContours(threshDV, 2, 1)
    cnt_DV = contours_DV[find_longest_cnt(contours_DV)]
    return cv2.matchShapes(cnt_DV, cnt_PWS, 1, 0.0)


def crop_image(reference, target):
    contours, _ = cv2.findContours(reference, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return target[y:y + h, x:x + w]


def padding_image(img, desired_size: list):
    resized_cropped_img_h, resized_cropped_img_w = img.shape[0], img.shape[1]
    pws_delta_h = desired_size[0] - resized_cropped_img_h
    pws_delta_w = desired_size[1] - resized_cropped_img_w
    pws_top, pws_bottom = pws_delta_h // 2, pws_delta_h - (pws_delta_h // 2)
    pws_left, pws_right = pws_delta_w // 2, pws_delta_w - (pws_delta_w // 2)
    return cv2.copyMakeBorder(img, pws_top, pws_bottom, pws_left, pws_right, cv2.BORDER_CONSTANT, None, value=0)


def img_show(inuput, title=""):
    plt.imshow(inuput)
    plt.title(title)
    plt.show()


def double_img_show(img1, img2, t1="", t2=""):
    plt.subplot(121)
    plt.title(t1)
    plt.imshow(img1)
    plt.subplot(122)
    plt.title(t2)
    plt.imshow(img2)
    plt.show()


def perspective_transform(ref_img, src_img, level_iters: list, inv_iter: int, visualization: bool = True):
    """
    Registering src_img to ref_img using Symmetric Diffeomorphic Registration algorithm
    :param ref_img: source image
    :param src_img: reference image for registration
    :param level_iters : list of int
            the number of iterations at each level of the Gaussian Pyramid (the
            length of the list defines the number of pyramid levels to be
            used)
    :param inv_iter : int
            the number of iterations to be performed by the displacement field
            inversion algorithm
    :param visualization: whether visualize the intermediate result
    :return: registered source image
    """

    def extract_mask(img):
        ret, thresh = cv2.threshold(img, 0.00001, img.max(), 0)
        thresh = np.uint8(thresh > 0) * 255
        return thresh

    ref_mask = extract_mask(ref_img)  # confocal mask (static)
    src_mask = extract_mask(src_img)  # pws mask (moving)

    # Registration setting
    metric = SSDMetric(ref_mask.ndim)  # select square difference matrix as similarity metric
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters=level_iters, inv_iter=inv_iter)
    mapping = sdr.optimize(ref_mask, src_mask)

    if visualization:
        regtools.overlay_images(ref_mask, src_mask, 'confocal mask', 'Overlay', 'pws mask')
        plt.show()
        regtools.plot_2d_diffeomorphic_map(mapping, 10)
        plt.show()
        warped_pwa_mask = mapping.transform(src_mask, 'linear')
        regtools.overlay_images(ref_mask, warped_pwa_mask, 'confocal mask', 'Overlay', 'Warped pws mask')
        plt.show()
        warped_pws = mapping.transform(src_img, 'linear')
        regtools.overlay_images(ref_img, warped_pws, 'confocal mask', 'Overlay', 'Warped pws')
        plt.show()

    return mapping.transform(src_img, 'linear')


class preprocess:
    def __init__(self,
                 PWS_Path: str,  # path to PWSimages folder, e.g. "BJCells_ACTD_08_26_2022/PWSimages"
                 DV_Path: str,  # path to DVimages folder, e.g. "BJCells_ACTD_08_26_2022/DVimages"
                 instance: str = ''):  # instance name

        # File path
        self.pws_path = os.path.join(PWS_Path, instance, "PWS/analyses/analysisResults_p0.h5")
        self.roi_path = os.path.join(PWS_Path, instance, "ROI_nuc.h5")
        self.dv_path = os.path.join(DV_Path, instance, "R3D_D3D-processed.tif")
        self.label_path = os.path.join(DV_Path, instance, "labels.mat")
        self.diagnose_txt = f"Processed/diagnose/{self.pws_path.split('/')[1]}.txt"

        # Save file name
        self.save_name = f"{DV_Path.split('/')[1]}_{instance}"

        # Raw data information
        self.pws = None
        self.dv = None
        self.dv_mask = None
        self.pws_mask = None

        # The index of all labeled confocal images, and their corresponding matching score
        self.label_index = None
        self.matching_score = None

        # Processed image
        self.pws_cell = None  # Masked pws cell image
        self.dv_info = None  # Dict contains various information relevant to dv cell
        self.cropped_pws = None  # Cropped pws image
        self.cropped_dv = None  # Cropped DV image
        self.padded_pws = None  # Padding the cropped DV image to [512, 512]
        self.padded_dv = None  # Padding the cropped DV image to [512, 512]
        self.registered_pws = None  # Registered pws image (refer to dv)

    def preprocessing_pipline(self, n: int,
                              output_size: list,
                              level_iters: list,
                              inv_iter: int,
                              norm_range: list = None,  # [pws.max-pws.min, dv.max-dv.min] (globally)
                              save: bool = False):
        """
        data preprocessing pipline
        :param n: int
                number of selected confocal images
        :param output_size: list of int
                the output size of both confocal/pws image
        :param level_iters : list of int
                the number of iterations at each level of the Gaussian Pyramid (the
                length of the list defines the number of pyramid levels to be
                used)
        :param inv_iter : int
                the number of iterations to be performed by the displacement field
                inversion algorithm
        :param norm_range: list of float
                a list contains [pws.max-pws.min, dv.max-dv.min]
        :param save: bool
                whether save the output images or not
        :return: None
        """
        self.read_file()  # return: self.pws, self.dv
        self.apply_mask_and_select(n=n, dialation=True)  # return: self.pws_cell, self.dv_info
        self.apply_cropping(resize=True)  # return: self.cropped_dv, self.cropped_pws
        # double_img_show(self.cropped_dv, self.cropped_pws, 'dv', 'pws')
        if norm_range is not None:
            self.normalize_data(norm_range)
            self.padding2same_size(desired_size=output_size)  # return: self.padded_dv, self.padded_pws
            self.image_registration(level_iters=level_iters, inv_iter=inv_iter, visualization=False)
            if save:
                regtools.overlay_images(self.padded_dv, self.registered_pws, 'confocal mask', 'Overlay', 'Warped pws',
                                        osp.join('./Processed/reference_img', f"{self.save_name}.png"),
                                        dpi=1000)
                plt.close('all')
                np.save(osp.join('./Processed/processed', f"{self.save_name}.npy"),
                        {'pws': self.registered_pws, 'confocal': self.padded_dv})

    def data_availability(self, instance, record: bool = True) -> bool:
        if not (os.path.exists(self.pws_path)):
            if record:
                with open(self.diagnose_txt, 'a') as f:
                    print(f"{instance} missing pws(analysisResults_p0.h5)", file=f)
            print(f"{instance} missing pws(analysisResults_p0.h5)")
            return False
        elif not (os.path.exists(self.roi_path)):
            if record:
                with open(self.diagnose_txt, 'a') as f:
                    print(f"{instance} missing ROI file", file=f)
            print(f"{instance} missing ROI file")
            return False
        elif not (os.path.exists(self.dv_path)):
            if record:
                with open(self.diagnose_txt, 'a') as f:
                    print(f"{instance} missing DV R3D_D3D-processed.tif", file=f)
            print(f"{instance} missing DV R3D_D3D-processed.tif")
            return False
        elif not (os.path.exists(self.label_path)):
            if record:
                with open(self.diagnose_txt, 'a') as f:
                    print(f"{instance} missing label.mat", file=f)
            print(f"{instance} missing label.mat")
            return False
        else:
            return True

    def read_file(self):
        # Raw PWS Image
        h5file = h5py.File(self.pws_path, 'r')
        self.pws = np.array(h5file['rms'])
        # ROI Mask
        h5file = h5py.File(self.roi_path, 'r')
        keys = [key for key in h5file.keys()]
        roi = h5file[keys[0]]['mask']
        self.pws_mask = np.array(roi)  # 1024×1024 from PWS mask
        # DV Image
        self.dv = io.imread(self.dv_path)  # tiff image series
        # Mask
        label = sio.loadmat(self.label_path)
        label = label['labels']
        self.dv_mask = np.transpose(label, (2, 0, 1))
        self.label_index = [i for i in range(self.dv_mask.shape[0]) if self.dv_mask[i].max() != 0]

    def apply_mask_and_select(self, n=3, dialation=True):
        self.pws_cell = np.multiply(self.pws, self.pws_mask.astype('float32'))
        self.dv_info = self.select_dv(n, dialation)
        return self.pws_cell, self.dv_info

    def apply_cropping(self, resize: bool = True):
        cropped_pws = crop_image(self.pws_mask, self.pws_cell)
        self.cropped_dv = crop_image(self.dv_info['avg_mask'], self.dv_info['avg_masked_cell'])
        if resize:
            self.cropped_pws = cv2.resize(cropped_pws, self.cropped_dv.shape[::-1])
        return self.cropped_dv, self.cropped_pws

    def normalize_data(self, norm_range):
        self.cropped_pws = self.cropped_pws / norm_range[0]
        self.cropped_dv = self.cropped_dv / norm_range[1]
        return self.cropped_dv, self.cropped_pws

    def padding2same_size(self, desired_size: list):
        self.padded_pws = padding_image(self.cropped_pws, desired_size)
        self.padded_dv = padding_image(self.cropped_dv, desired_size)
        return self.padded_dv, self.padded_pws

    def image_registration(self, level_iters: list, inv_iter: int, visualization: bool):
        assert self.padded_dv is not None and self.padded_pws is not None
        self.registered_pws = perspective_transform(ref_img=self.padded_dv,
                                                    src_img=self.padded_pws,
                                                    level_iters=level_iters,
                                                    inv_iter=inv_iter,
                                                    visualization=visualization)
        return self.registered_pws

    def select_dv(self, n: int = 3, dialation: bool = True) -> dict:
        self.matching_score = [get_matching_score(self.pws_mask, self.dv_mask[idx]) for idx in self.label_index]

        def find_DV_ind(matching_score, n: int = 3) -> int:
            """find the 3 least value in the output score list"""
            assert len(matching_score) >= 3
            step_len = len(matching_score) - 2
            start_index = np.argmin([sum(matching_score[step:step + n]) / n for step in range(step_len)])
            return start_index

        window_min = find_DV_ind(self.matching_score, n=n)
        dv_mask = [self.dv_mask[select_ind].astype('float32') for select_ind in
                   self.label_index[window_min:window_min + n]]
        avg_dv_mask = np.average(dv_mask, axis=0)
        avg_dv_mask[avg_dv_mask > 0] = 1
        if dialation:
            avg_dv_mask = avg_dv_mask.astype(np.uint8).copy()
            avg_dv_mask = cv2.dilate(avg_dv_mask, np.ones((2, 2), np.uint8), iterations=3)
        selected_dv_cell = [np.multiply(self.dv[select_ind], avg_dv_mask.astype('float32'))
                            for select_ind in self.label_index[window_min:window_min + n]]
        avg_cell = np.average(selected_dv_cell, axis=0)
        return {'selected_dv': selected_dv_cell, 'avg_masked_cell': avg_cell, 'avg_mask': avg_dv_mask}


if __name__ == '__main__':
    pass
