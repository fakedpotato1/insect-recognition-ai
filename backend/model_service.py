from config import Config

import mock_model
import real_model 


def predict(image):

    if Config.USE_MOCK_MODEL:
        return mock_model.predict(image)

    return real_model.predict(image)