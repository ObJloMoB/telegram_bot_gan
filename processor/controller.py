from threading import Thread
from collections import deque
from time import sleep

import torch 
import torchvision.transforms as transforms
import numpy as np
import cv2
import os

from model import Generator

from time import time
def timer(func):
    def wrap(*args, **kwargs):
        t_start = time()
        retval = func(*args, **kwargs)
        print(f'[{func.__name__}] processing time: {(time()-t_start)*1000:0.4f} milis')
        return retval
    return wrap


class Controller:
    def __init__(self, 
                 weights='weights/Shinkai_net_G_float.pth', 
                 img_size=400, 
                 async=False):
        self.model = Generator()
        full_weights = os.path.join(os.path.dirname(__file__), weights)
        self.model.load_state_dict(torch.load(full_weights))
        self.model.eval()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.size = img_size

        if async:
            self.in_queue = deque()
            self.in_dict = dict()
            self.out_dict = dict()

            self.started = True
            self.processing_thread = Thread(target=self.processing)
            self.processing_thread.start()

    def set_async_req(self, key, image):
        self.in_queue.append(key)
        self.in_dict.update({key: image})

    def get_asyc_resp(self, key):
        if key in self.out_dict.keys():
            return self.out_dict.pop(key)
        else:
            return None

    def processing(self):
        while self.started:
            try:
                key = self.in_queue.popleft()
                image = self.in_dict.pop(key)

                res_image = self.process_image(image)
                self.out_dict.update({key: res_image})
            except (KeyError, IndexError):
                sleep(0.1)

    @timer
    def process_image(self, input_image):
        # Thumbnailing
        inp_w = input_image.shape[0]
        inp_h = input_image.shape[1]
        ratio = inp_h *1.0 / inp_w
        if ratio > 1:
            h = self.size 
            w = int(h*1.0/ratio)
        else:
            w = self.size 
            h = int(w * ratio)
        input_image = cv2.resize(input_image, (h, w), cv2.INTER_CUBIC)

        # Cast to tensor
        input_image = transforms.ToTensor()(input_image).to(self.device).unsqueeze(0)

        # Weird for me preprocessing
        input_image = -1 + 2 * input_image

        # Processing
        output_image = self.model(input_image)

        # Recreate opencv image
        output_image = output_image[0]
        output_image = output_image.data.cpu().float() * 0.5 + 0.5
        output_image = output_image.numpy().transpose((1, 2, 0)) * 255
        output_image = output_image.astype(np.uint8)
        return output_image

    def stop(self):
        self.started = False
        self.processing_thread.join()


if __name__ == '__main__':
    in_fld = 'folder_with_images_blush'
    import os

    async = True
    contr = Controller(async=async)
    if async:
        for name in os.listdir(in_fld):
            img = cv2.imread(os.path.join(in_fld, name))
            contr.set_async_req(name, img)


        met_names = list()
        while True:
            for name in os.listdir(in_fld):
                if name not in met_names:
                    resp = contr.get_asyc_resp(name)
                    if resp is not None:
                        met_names.append(name) 
                        cv2.imshow('r', resp)
                        cv2.waitKey(0)
                    else:
                        print('no upds')
                        sleep(0.1)
    else:
        for name in os.listdir(in_fld):
            img = cv2.imread(os.path.join(in_fld, name))
            res_img = contr.process_image(img)
            cv2.imshow('r', res_img)
            cv2.waitKey(0)


