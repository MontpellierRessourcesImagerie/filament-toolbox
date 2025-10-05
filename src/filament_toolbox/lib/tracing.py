import numpy as np
from brightest_path_lib.algorithm import NBAStarSearch
from brightest_path_lib.algorithm import AStarSearch


class BrightestPathTracing(object):


    def __init__(self, image, points):
        super().__init__()
        self.image = image
        self.points = points
        self.methods=["A-star", "NBA-star"]
        self.method_text = "NBA-star"
        self.methods = {"A-star": AStarSearch, "NBA-star": NBAStarSearch}
        self.result = None


    def run(self):
        self.result = np.zeros(self.image.shape, np.uint16)
        method = self.methods[self.method_text]
        for index in range(len(self.points)-1):
            start = self.points[index]
            end = self.points[index + 1]
            algorithm = method(self.image, start, end)
            path = algorithm.search()
            for z, y, x in path:
                self.result[z][y][x] = index + 1