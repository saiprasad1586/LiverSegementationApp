from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import CacheDataset, DataLoader, Dataset

import torch
import matplotlib.pyplot as plt

import os
from glob import glob
import numpy as np
import cv2

from tqdm import tqdm
from monai.inferers import sliding_window_inference


class Predict:

    def __init__(self, patient_id, slice_location):
        self.patient_id = str(patient_id)
        self.slice_location = slice_location
        # self.
        self.device = torch.device("cuda:0")
        self.model_path = 'results/results/best_metric_model.pth'
        self.model = self.load_model()
        self.data_loader = self.load_data()
        self.result_img_location = self.make_result_patient_img()
        self.result_seg_location = self.make_result_patient_seg()
        self.final_result_location = self.make_final_result()

    def load_model(self):
        print("[INFO] Loading model")
        model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        # print(self.model_path)
        model.eval()
        print("[INFO] Loading model complete")
        return model

    def load_data(self):
        print("[INFO] Loading Data")
        # files = [{"vol": image_name} for image_name in zip(glob(self.slice_location))]
        slice_location = glob(self.slice_location)
        # print(slice_locatioin)
        files = [{"vol": image_name} for image_name in zip(slice_location)]
        test_transforms = Compose([
            LoadImaged(keys=["vol"]),
            AddChanneld(keys=["vol"]),
            # Spacingd(keys=["vol"], pixdim=(1.5,1.5,1.0), mode=("bilinear")),
            Orientationd(keys=["vol"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['vol'], source_key='vol'),
            Resized(keys=["vol"], spatial_size=[256, 256, 64]),
            ToTensord(keys=["vol"]),
        ])
        # print(self.test_transforms)

        # print(files)
        test_ds = CacheDataset(data=files, transform=test_transforms)
        data_loader = DataLoader(test_ds, batch_size=1)
        # print(first(data_loader))
        print("[INFO] Loading Data Complete")
        return data_loader

    def make_result_patient_img(self):
        os.mkdir(os.path.join('result_img', self.patient_id))
        os.mkdir(os.path.join('result_img', self.patient_id, 'images'))
        return os.path.join('result_img', self.patient_id, 'images')

    def make_result_patient_seg(self):
        os.mkdir(os.path.join('result_img', self.patient_id, 'seg'))
        return os.path.join('result_img', self.patient_id, 'seg')

    def make_final_result(self):
        os.mkdir(os.path.join('static/result', self.patient_id))
        return os.path.join('static/result', self.patient_id)

    def predict_on_slice(self):

        # for test_patient in iter(self.data_loader):

        sw_batch_size = 4
        roi_size = (256, 256, 64)
        with torch.no_grad():
            test_patient = first(self.data_loader)
            vol = test_patient['vol']

            test_output = sliding_window_inference(vol.to(self.device), roi_size, sw_batch_size, self.model)
            sigmoid_activation = Activations(sigmoid=True)
            test_outputs = sigmoid_activation(test_output)
            test_outputs = test_outputs > 0.53

            for i in tqdm(range(64)):
                plt.figure("check",(18, 6))
                plt.title(f"image {i}")
                plt.imshow(test_patient["vol"][0, 0, :, :, i], cmap="gray")
                plt.savefig(self.result_img_location + '/image{}.png'.format(i))
                # plt.show()

                plt.figure("check", (18, 6))
                plt.title(f"Predicted output {i}")
                plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, i], cmap="gray")
                plt.savefig(self.result_seg_location + '/image{}.png'.format(i))

    def add_colored_mask(self, image, mask_image):
        mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

        mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)

        mask_coord = np.where(mask != [0, 0, 0])

        mask[mask_coord[0], mask_coord[1], :] = [255, 0, 0]

        ret = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

        return ret

    def final_result(self):
        imgs = sorted(glob(self.result_img_location+'/*'))
        segs = sorted(glob(self.result_seg_location+'/*'))

        files = [{"img": image_name, "seg": seg_name} for image_name, seg_name in zip(imgs, segs)]
        for i in tqdm(range(64)):
            image = cv2.imread(files[i]['img'])
            # print(files[i]['seg'])
            mask_image = cv2.imread(files[i]['seg'])
            merged_img = self.add_colored_mask(image, mask_image)
            merged_image = merged_img[75:520, 692:1155]
            cv2.imwrite(os.path.join(self.final_result_location, "image{}.png".format(i)), merged_image)

    def return_final_path(self):
        print(self.slice_location)
        self.predict_on_slice()
        self.final_result()
        return self.final_result_location

if __name__ == '__main__':
    predict = Predict('13', 'Uploads/liver_16_6.nii.gz')
    predict.predict_on_slice()
    predict.final_result()
