class CvFunction:
    @staticmethod
    def process(image, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_params():
        raise NotImplementedError
