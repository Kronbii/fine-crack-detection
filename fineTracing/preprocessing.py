from . import utils
from . import edge
import numpy as np


def main(image) -> np.ndarray:
    #processed_img = utils.opening(image)
    processed_img = edge.main(image)
    # processed_img = utils.GaussianBlur(processed_img, 5)
    return processed_img


if __name__ == "__main__":
    main()
