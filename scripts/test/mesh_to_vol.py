from resector.io import read_poly_data, write
from resector.mesh import mesh_to_volume

mesh_path = '/tmp/noisy_poly_data.vtp'
reference_path = '/home/fernando/Desktop/resector_test_image/1395_resectable_left_label.nii.gz'

poly_data = read_poly_data(mesh_path)

image = mesh_to_volume(poly_data, reference_path)
write(image, '/tmp/stenciled.nii.gz')
