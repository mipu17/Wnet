import random
import math

class Node:

    def __init__(self, id, xmax, ymax, xmin, ymin):
        self.x_axis = random.uniform(xmin, xmax)
        self.y_axis = random.uniform(ymin, ymax)
        self.id = id
        self.location = self.x_axis, self.y_axis

    def to_dict(self):
        return {
            'x_axis': self.x_axis,
            'y_axis': self.y_axis,
            'id':self.id,
        }

    def NodeClusterDist(self, cx, cy, radius):
        inside = False
        distance = 0
        distance = math.sqrt(((cx - self.x_axis) ** 2) + ((cy - self.y_axis) ** 2))
        if distance < radius:
            inside = True

        return inside, distance



