class SubtractImage:

    def __init__(self, image1, image2):
        self.image1 = image1
        self.image2 = image2
        self.result = None

    def run(self):
        self.result = self.image1 - self.image2
