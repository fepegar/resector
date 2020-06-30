from pathlib import Path
import numpy as np
import sitkUtils as su
from ScreenCapture import ScreenCaptureLogic


def jump(r, a, s):
    colors = 'Yellow', 'Green', 'Red'
    center = r, a, s
    for (color, offset) in zip(colors, center):
        sliceLogic = slicer.app.layoutManager().sliceWidget(color).sliceLogic()
        sliceLogic.SetSliceOffset(offset)


images_dir = Path('/tmp/resector/patient')
screenshots_dir = Path('/tmp/screenshots')
screenshots_dir.mkdir(exist_ok=True)

slicer.mrmlScene.Clear(0)

volumeNode = loadVolume(str(images_dir / '0713_t1_pre_on_mni.nii.gz.nii.gz'))
labelNode = loadLabelVolume(str(images_dir / '0713_t1_pre_NeuroMorph_Parcellation.nii.gz'))
loadGIFColorTable()
colorNode = getNode('BrainAnatomyLabelsV3_0')
displayNode = labelNode.GetDisplayNode()
displayNode.SetAndObserveColorNodeID(colorNode.GetID())

logic = ScreenCaptureLogic()
logic.showViewControllers(False)

# jump(-2.68, 7.3, 9.2)

setSliceViewerLayers(background=volumeNode, label=None)
slicer.app.processEvents()
logic.captureImageFromView(None, str(screenshots_dir / 'initial_mri.png'))

setSliceViewerLayers(background=None, label=labelNode)
slicer.app.processEvents()
logic.captureImageFromView(None, str(screenshots_dir / 'initial_gif.png'))




for i in range(10, 61):
    slicer.mrmlScene.Clear(0)
    seg_path, mri_path = sorted(list(images_dir.glob(f'*_{i}.nii.gz')))
    print(seg_path)
    print(mri_path)
    volumeNode = loadVolume(str(mri_path))
    print(volumeNode.GetName())
    labelNode = loadLabelVolume(str(seg_path))
    segNode = loadSegmentation(str(seg_path))
    a = array(labelNode.GetName())
    index = np.array(np.where(a)).mean(axis=1)[::-1]  # numpy to itk
    labelImage = su.PullVolumeFromSlicer(labelNode.GetName())
    l, p, s = labelImage.TransformContinuousIndexToPhysicalPoint(index)
    r, a = -l, -p
    jump(r, a, s)
    slicer.mrmlScene.RemoveNode(labelNode)
    screenshot_path = screenshots_dir / mri_path.name.replace('.nii.gz', '_label.png')
    logic.captureImageFromView(None, str(screenshot_path))
    displayNode = segNode.GetDisplayNode()
    displayNode.SetVisibility(False)
    screenshot_path = screenshots_dir / mri_path.name.replace('.nii.gz', '_resected.png')
    slicer.app.processEvents()
    logic.captureImageFromView(None, str(screenshot_path))
    print()

logic.showViewControllers(True)




# Outside slicer
from skimage import io
from pathlib import Path
import numpy as np

screenshots_dir = Path('/tmp/screenshots')
frames_dir = screenshots_dir / 'frames'
frames_dir.mkdir(exist_ok=True)

for i in range(1, 61):
    label_path, resected_path = sorted(list(screenshots_dir.glob(f'*_{i}_*.png')))
    resected = io.imread(resected_path)
    label = io.imread(label_path)
    frame = np.hstack((resected, label))
    frame_path = frames_dir / f'frame_{i:02}.png'
    io.imsave(frame_path, frame)


# From terminal
# $ convert -resize 50% -delay 200 -loop 0 *.png ../output_50.gif
