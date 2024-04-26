class Data:
    def __init__(self, image, contours, class_name, probs):
        self._image = image
        self._contours = contours
        self._class_name = class_name
        self._probs = probs

    @property
    def image(self):
        return self._image

    @property
    def contours(self):
        return self._contours

    @property
    def class_name(self):
        return self._class_name

    @property
    def probs(self):
        return self._probs