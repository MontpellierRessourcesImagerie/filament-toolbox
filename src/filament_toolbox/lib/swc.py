import numpy as np



class Node(object):


    def __init__(self, filament_id, filament_type, coordinates, radius, parent):
        super().__init__()
        self.id = filament_id
        self.type = filament_type
        self.coords = coordinates
        self.radius = radius
        self.parent = parent



class SWCForest(object):


    def __init__(self, data):
        super().__init__()
        self.data = data


    @classmethod
    def read_from(cls, paths):
        filaments = []
        for path in paths:
            filament = {}
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
                x = float(columns[2])
                y = float(columns[3])
                z = float(columns[4])
                radius = float(columns[5])
                parent = int(columns[6])
                filament[id] = Node(f_id, f_type, (z, y, x), radius, parent)
            filaments.append(filament)
        data = []
        for filament in filaments:
            for nodeID, node in filament.items():
                parent = node.parent
                if parent == -1:
                    continue
                parent_node = filament[parent - 1]
                data.append(np.array([node.coords, parent_node.coords]))
        return SWCForest(data)