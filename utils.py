import cv2 as cv

from resnet_152 import resnet152_model
from resnet_50 import resnet50_model

from classification_models.tfkeras import Classifiers
ResNet18, preprocess_input = Classifiers.get('resnet18')

def load_model():
    model_weights_path = 'models/model.42-0.88.hdf5'
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 203
    model = resnet50_model(img_height, img_width, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)
    # model = ResNet18((224, 224, 3), weights="models/model.12-0.82.hdf5", classes = 203)
    return model


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
