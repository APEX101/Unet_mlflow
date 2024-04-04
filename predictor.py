from tensorflow.keras.models import load_model
from data import *
from model import unet
from logger import logger
import argparse
import cv2

# changes to test workflow and more test
parser = argparse.ArgumentParser(description="Predict a UNet model on membrane data.")
parser.add_argument(
    "--image_path", type=str, default="nearest", help="Fill mode for data augmentation"
)



class Infer:
    def __init__(self, chkpt_path=None) -> None:
        # :TODO load the model using tf
        if chkpt_path:

            self.model = load_model(chkpt_path)
        self.model = unet()

    def predictor(self, input, thresh):
        # perform preporcess on the data

        # get predictions
        prob_mask = self.model(input)

        # perform post process
        mask = np.where(prob_mask > thresh, 255, 0)

        return mask


if __name__ == "__main__":
    cl = Infer("unet_membrane.keras")
    # testGene = testGenerator("data/membrane/test")
    # inst = next(iter(testGene))
    args = parser.parse_args()
    inst = cv2.imread(args.image_path)
    pred = cl.predictor(inst)
    logger.info(f"Prediction shape {pred.shape}")
