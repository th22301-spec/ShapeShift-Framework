from abaqus import *
from abaqusConstants import *
from part import *
from material import *
from section import *
from assembly import *
from step import *
from load import *
from mesh import *
from job import *
import numpy as np

# Get user inputs for the keyed shaft dimensions
inputs = getInputs(
    fields=[
        ('Enter shaft diameter (mm):', '10.0'),
        ('Enter keyway width (mm):', '2.0'),
        ('Enter keyway height (mm):', '1.0'),
        ('Enter angle of twist (Deg):', '0.5'),
        ('Enter mesh size (mm):', '0.5')
    ],
    label='Keyed Shaft Dimensions'
)

d = float(inputs[0])  # Shaft diameter
w = float(inputs[1])  # Keyway width
h = float(inputs[2])  # Keyway height
angle = float(inputs[3])
angle = np.radians(angle)
r = d/2
keypoint1 = (-w/2,np.sqrt(d**2 - w**2) / 2)
keypoint2 = (-w/2,r-h)
keypoint3 = (w/2,r-h)
keypoint4 = (w/2,np.sqrt(d**2 - w**2) / 2)
mesh_size = float(inputs[4])  # Mesh size
length = 50  # Fixed shaft length

# Create a new model
model = mdb.models['Model-1']

# Create the shaft with a circular cross-section
sketch = model.ConstrainedSketch(name='ShaftSketch', sheetSize=20.0)
sketch.Line(point1=keypoint1, point2=keypoint2)
sketch.Line(point1=keypoint2, point2=keypoint3)
sketch.Line(point1=keypoint3, point2=keypoint4)
sketch.ArcByCenterEnds(center=(0.0, 0.0), point1=keypoint1, point2=keypoint4, direction=COUNTERCLOCKWISE)
shaft_part = model.Part(name='KeyedShaft', dimensionality=THREE_D, type=DEFORMABLE_BODY)
shaft_part.BaseSolidExtrude(sketch=sketch, depth=length)
#del model.sketches['ShaftSketch']

# Define the material properties
material_name = 'PLAsolid'
model.Material(name=material_name)
material = model.materials[material_name]
material.Elastic(table=((3250.0, 0.3),))  # E=3250 MPa, v=0.3
material.Plastic(table=((48.57, 0.0), (52.5, 0.019), (45.5, 0.063)))  # Yield stress and plastic strain

# Create a solid section and assign the material
section_name = 'ShaftSection'
model.HomogeneousSolidSection(name=section_name, material=material_name, thickness=None)

# Assign the section to the keyed shaft
region = shaft_part.Set(cells=shaft_part.cells, name='ShaftRegion')
shaft_part.SectionAssignment(region=region, sectionName=section_name)

# Create the assembly
assembly = model.rootAssembly
assembly.DatumCsysByDefault(CARTESIAN)

# Add the keyed shaft to the assembly
shaft_instance = assembly.Instance(name='KeyedShaftInstance', part=shaft_part, dependent=ON)

# Verify the instance
print("Keyed shaft added to the assembly as an instance.")
# Rotate the keyed shaft instance
assembly.rotate(
    instanceList=('KeyedShaftInstance',),  # Name of the instance to rotate
    axisPoint=(0.0, 0.0, 0.0),             # Center of rotation
    axisDirection=(0.0, 1.0, 0.0),         # Rotation along the Y-axis
    angle=90.0                             # Rotation angle in degrees
)

# Verify the rotation
print("Keyed shaft instance rotated 90 degrees around the Y-axis.")
# Create a reference point at (length, 0, 0)
rp_coordinates = (length, 0.0, 0.0)  # Specify the coordinates of the RP
reference_point = assembly.ReferencePoint(point=rp_coordinates)

# Get the reference point ID for future use
rp_id = reference_point.id

# Create a set for the reference point
assembly.Set(referencePoints=(assembly.referencePoints[rp_id],), name='RP_Set')

# Verify the reference point creation
# Define a static general step with initial increment size of 0.01
step_name = 'StaticStep'
model.StaticStep(
    name=step_name,
    previous='Initial',   # The step follows the default initial step
    description='Static general step for applying loads and boundary conditions.',
    nlgeom=ON,            # Enable nonlinear geometry if necessary
    initialInc=0.01,      # Set the initial increment size
    maxInc=0.1,           # Optional: Set a maximum increment size
    maxNumInc=1000        # Optional: Set a maximum number of increments
)

# Verify the step creation
regionDef=mdb.models['Model-1'].rootAssembly.sets['RP_Set']
mdb.models['Model-1'].HistoryOutputRequest(name='H-Output-2', 
        createStepName='StaticStep', variables=('UR1', 'RM1'), 
        region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)

# Select the face of the keyed shaft at (length, 0, 0)
shaft_face = shaft_instance.faces.findAt(((length, 0.0, 0.0),))

# Create a surface set for the coupling constraint
assembly.Surface(side1Faces=shaft_face, name='CouplingSurface')

# Create the coupling constraint
coupling_name = 'RP_Coupling'
mdb.models['Model-1'].Coupling(
    name=coupling_name,
    controlPoint=assembly.sets['RP_Set'],  # Reference point set
    surface=assembly.surfaces['CouplingSurface'],  # Surface set
    influenceRadius=WHOLE_SURFACE,  # Coupling applies to the whole surface
    couplingType=KINEMATIC,        # Use kinematic coupling
    u1=ON, u2=ON, u3=ON,           # Constrain translations
    ur1=ON, ur2=ON, ur3=ON         # Constrain rotations
)


# Verify the coupling creation

mdb.models['Model-1'].DisplacementBC(name='BC-RefPoint', createStepName='Initial', region=assembly.sets['RP_Set'], u1=0, u2=0, u3=0, ur1=0, ur2=0, ur3=0)

mdb.models['Model-1'].DisplacementBC(name='BC-RefPoint', createStepName='StaticStep', region=assembly.sets['RP_Set'], u1=0, u2=0, u3=0, ur1=angle, ur2=0, ur3=0)

# Select the face of the keyed shaft at (0, 0, 0)
encastre_face = shaft_instance.faces.findAt(((0.0, 0.0, 0.0),))

# Create a surface set for the Encastre boundary condition
assembly.Surface(side1Faces=encastre_face, name='EncastreSurface')

# Apply the Encastre boundary condition
assembly.Set(name='EncastreSet', faces=encastre_face)
model.EncastreBC(name='BC-Encastre', createStepName='Initial', region=assembly.sets['EncastreSet'])

# Verify the Encastre boundary condition
print("Encastre boundary condition applied to the surface at (0, 0, 0).")

# Define the mesh size
shaft_part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)

# Assign the mesh controls
shaft_part.setMeshControls(regions=shaft_part.cells, technique=SWEEP)

# Generate the mesh
shaft_part.generateMesh()

# Verify the meshing

# Define the job name
job_name = 'KeyedShaftJob'

# Create the job
mdb.Job(
    name=job_name,
    model='Model-1',
    description='Analysis job for the keyed shaft with specified boundary conditions and loads.',
    type=ANALYSIS,
    explicitPrecision=SINGLE,
    nodalOutputPrecision=SINGLE,
    multiprocessingMode=DEFAULT,
)