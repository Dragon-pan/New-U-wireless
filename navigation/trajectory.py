import numpy as np
from scipy.sparse.csgraph import shortest_path


class Trajectory():
    def __init__(self,map_data):
        self.lines=map_data['lines']
        self.M=map_data['graph']
        self.anchor_name=map_data['anchor_name']
        self.anchor_location=map_data['anchor_location']

    def ccw(self,A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def distance(self,c, d):
        for boundary in self.lines:
            a, b = [boundary[0], boundary[1]], [boundary[2], boundary[3]]
            if self.ccw(a, c, d) != self.ccw(b, c, d) and self.ccw(a, b, c) != self.ccw(a, b, d):
                return 0
        return np.linalg.norm(np.array(c) - np.array(d))

    def calculate_path(self,pose,destination_id):
        for i,loc in enumerate(self.anchor_location):
            self.M[i,-1]=self.distance(pose, loc)
        _, Pr = shortest_path(self.M, directed=True, method='FW', return_predecessors=True)
        index=self.anchor_name.index(destination_id)
        Pr=Pr[index]
        path_list=[]
        index=Pr[-1]
        while index!=-9999:
            path_list.append(self.anchor_location[index])
            index=Pr[index]
        return path_list