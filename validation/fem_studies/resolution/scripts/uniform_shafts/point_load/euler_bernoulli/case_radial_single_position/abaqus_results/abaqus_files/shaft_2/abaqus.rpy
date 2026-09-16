# -*- coding: mbcs -*-
#
# Abaqus/Viewer Release 2025 replay file
# Internal Version: 2024_09_20-14.00.46 RELr427 198590
# Run by jmrap on Wed Sep  9 18:33:49 2026
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=195.515625, 
    height=108.78588104248)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from viewerModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
o2 = session.openOdb(name='shaft_2_euler.odb')
#: Model: C:/Users/jmrap/Documents/GitHub/AxisForge-Design-Studies/validation/fem_studies/resolution/scripts/uniform_shafts/point_load/euler_bernoulli/case_radial_single_position/abaqus_results/abaqus_files/shaft_2/shaft_2_euler.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       1
#: Number of Node Sets:          1
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o2)
session.viewports['Viewport: 1'].makeCurrent()
session.Path(name='Path-1', type=NODE_LIST, expression=(('SHAFT_1-1', (1, 10, 
    )), ))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'))
xyp = session.XYPlot('XYPlot-1')
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
pth = session.paths['Path-1']
xy1 = xyPlot.XYDataFromPath(path=pth, includeIntersections=True, 
    projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, 
    projectionTolerance=0, shape=UNDEFORMED, labelType=TRUE_DISTANCE_X, 
    removeDuplicateXYPairs=True, includeAllElements=False)
c1 = session.Curve(xyData=xy1)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
pth = session.paths['Path-1']
session.XYDataFromPath(name='XYData-U2', path=pth, includeIntersections=True, 
    projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, 
    projectionTolerance=0, shape=UNDEFORMED, labelType=TRUE_DISTANCE_X, 
    removeDuplicateXYPairs=True, includeAllElements=False)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U3'))
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
pth = session.paths['Path-1']
xy1 = xyPlot.XYDataFromPath(path=pth, includeIntersections=True, 
    projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, 
    projectionTolerance=0, shape=UNDEFORMED, labelType=TRUE_DISTANCE_X, 
    removeDuplicateXYPairs=True, includeAllElements=False)
c1 = session.Curve(xyData=xy1)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
pth = session.paths['Path-1']
session.XYDataFromPath(name='XYData-U3', path=pth, includeIntersections=True, 
    projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, 
    projectionTolerance=0, shape=UNDEFORMED, labelType=TRUE_DISTANCE_X, 
    removeDuplicateXYPairs=True, includeAllElements=False)
xyp = session.xyPlots['XYPlot-1']
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
pth = session.paths['Path-1']
xy1 = xyPlot.XYDataFromPath(path=pth, includeIntersections=True, 
    projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, 
    projectionTolerance=0, shape=UNDEFORMED, labelType=TRUE_DISTANCE_X, 
    removeDuplicateXYPairs=True, includeAllElements=False)
c1 = session.Curve(xyData=xy1)
chart.setValues(curvesToPlot=(c1, ), )
session.charts[chartName].autoColor(lines=True, symbols=True)
import sys
sys.path.insert(8, 
    r'c:/SIMULIA/EstProducts/2025/win_b64/code/python3.10/lib/abaqus_plugins/excelUtilities')
import abq_ExcelUtilities.excelUtilities
abq_ExcelUtilities.excelUtilities.XYtoExcel(xyDataNames='XYData-U2,XYData-U3', 
    trueName='From Current XY Plot')
#: Multiple XY Data are exported. No chart will be created.
#: XY Data sent to Excel
