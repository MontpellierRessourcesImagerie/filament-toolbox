import os
import numpy as np
from skimage.draw import line_nd
from skimage.measure import label
from skan import Skeleton


class Node(object):


    def __init__(self, filament_id, filament_type, coordinates, radius, parent):
        super().__init__()
        self.id = filament_id
        self.type = filament_type
        self.coords = coordinates
        self.radius = radius
        self.parent = parent



class SWCForest(object):


    def __init__(self, data, scale=(1,1,1)):
        super().__init__()
        self.data = data
        self.scale = scale


    def get_shape(self):
        data = np.transpose(np.array(self.data))
        shape = int(round(np.max(data[0])))+1, int(round(np.max(data[1])))+1, int(round(np.max(data[2])))+1
        return shape


    def get_skeleton(self):
        mask = np.zeros(self.get_shape())
        for data in self.data:
            indices = line_nd(data[0], data[1], integer=True)
            mask[indices] = 255
        labels = label(mask)
        skeleton = Skeleton(labels, spacing=self.scale)
        return skeleton


    @classmethod
    def read_from(cls, paths, scale=(1,1,1)):
        print("paths:", paths)
        filaments, names = cls._read_filaments_from(paths)
        data = []
        for filament in filaments:
            for node in filament:
                parent = node.parent
                if parent == -1:
                    continue
                parent_node = filament[parent - 1]
                if node.coords == parent_node.coords:
                    continue
                data.append(np.array([node.coords, parent_node.coords]))
        forest = SWCForest(data, scale=scale)
        forest.name = names[0]
        return forest


    @classmethod
    def _read_filaments_from(cls, paths):
        filaments = []
        names = []
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            names.append(name)
            filament = []
            lines = None
            with open(path, "r") as f:
                lines = f.readlines()
            if not lines:
                continue
            for line in lines:
                if line.strip().startswith('#'):
                    continue
                columns = line.strip().split(" ")
                f_id = int(columns[0])
                f_type = int(columns[1])
                x = int(round(float(columns[2])))
                y = int(round(float(columns[3])))
                z = int(round(float(columns[4])))
                radius = float(columns[5])
                parent = int(columns[6])
                filament.append(Node(f_id, f_type, (z, y, x), radius, parent))
            filaments.append(filament)
        return filaments, names