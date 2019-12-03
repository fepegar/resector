"""
Generate a geodesic sphere

1) Clone the Python 3 fork of pyDome: https://github.com/fepegar/pyDome
2) Run python pyDome.py -o test -r 0.5 -f 16 -F > /dev/null

Step 2 creates a file test.wrl with 2562 points and 5120 triangles

Then run this code.
"""


import re
from pathlib import Path

import vtk
import vtk.util.numpy_support as vtk_np
import numpy as np


input_path = Path('test.wrl')
output_path = '/tmp/geodesic_polyhedron.vtp'



text = input_path.read_text()

pattern = r'point \[([\s\S]+?)\]'  # https://stackoverflow.com/a/33312193/3956024
lines = re.findall(pattern, text)[0].splitlines()
points = []
for line in lines:
    point = [float(n) for n in line.rstrip(',').split()]
    if not point: continue
    points.append(point)
nparray = np.array(points)



pattern = r'coordIndex \[([\s\S]+?)\]'  # https://stackoverflow.com/a/33312193/3956024
lines = re.findall(pattern, text)[0].splitlines()
cells = []
for line in lines:
    split = line.split(',')
    try:
        cell = [int(n) for n in split if n]
        cells.append(cell)
    except Exception:
        continue
cells_array = np.array(cells)
cells_array = cells_array[:, :-1]


def getMesh(vertices, faces):
    # http://www.vtk.org/Wiki/VTK/Examples/Python/DataManipulation/Cube.py
    def mkVtkIdList(it):
        vil = vtk.vtkIdList()
        for i in it:
            vil.InsertNextId(int(i))
        return vil

    polyData = vtk.vtkPolyData()
    pointsObject = vtk.vtkPoints()
    facesObject = vtk.vtkCellArray()

    for i in range(len(vertices)):
        pointsObject.InsertPoint(i, vertices[i])
    for i in range(len(faces)):
        facesObject.InsertNextCell(mkVtkIdList(faces[i]))

    polyData.SetPoints(pointsObject)
    polyData.SetPolys(facesObject)

    return polyData

pd = getMesh(nparray, cells_array)


normals = vtk.vtkPolyDataNormals()
normals.SetInputData(pd)


writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(output_path)
writer.SetInputConnection(normals.GetOutputPort())
writer.Write()
