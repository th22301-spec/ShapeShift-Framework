import numpy as np
from abaqus import *
from abaqusConstants import *
import os
import section
import regionToolset
import part
import material
import assembly
import step
import mesh
import job
import sketch

def read_gcode():
    os.chdir(r"C:\Users\th22301\OneDrive - University of Bristol\Desktop\Abaqus-python-practice")
    file = getInputs(fields=[('Enter G-code file name', '')])[0].strip().lower()
    gcode_filename = file + '.gcode'

    with open(gcode_filename, 'r') as f:
        data = f.readlines()

    sentences = [line.strip() for line in data if line.strip()]

    nozzle_diameter = None
    for sentence in sentences:
        if 'NOZZLE.DIAMETER' in sentence:
            nozzle_diameter = float(sentence.split()[-1][35:37])
            nozzle_diameter = nozzle_diameter + 0.02
            break

    print("Nozzle diameter = " + str(nozzle_diameter) + " mm.")

    layer_height = None
    for sentence in sentences:
        if ';PRINT.SIZE.MIN.Z' in sentence:
            layer_height = float(sentence.split()[-1][19:21])
            break

    print("Layer height = " + str(layer_height) + " mm.")

    print_commands = [s for s in sentences if 'G0' in s or 'G1' in s]
    print_commands_XYZ = [cmd for cmd in print_commands if 'X' in cmd or 'Y' in cmd or 'Z' in cmd]

    g, x, y, z = [], [], [], []

    for cmd in print_commands_XYZ:
        cmd_split = cmd.split()
        g_val = int(cmd_split[0].replace('G', ''))
        g.append(g_val)

        x_val, y_val, z_val = np.nan, np.nan, np.nan
        for item in cmd_split:
            if 'X' in item:
                x_val = float(item.replace('X', ''))
            if 'Y' in item:
                y_val = float(item.replace('Y', ''))
            if 'Z' in item:
                z_val = float(item.replace('Z', ''))

        x.append(x_val)
        y.append(y_val)
        z.append(z_val)

    for i in range(1, len(x)):
        if np.isnan(x[i]):
            x[i] = x[i - 1]
        if np.isnan(y[i]):
            y[i] = y[i - 1]
        if np.isnan(z[i]):
            z[i] = z[i - 1]

    event_series = [{'G': g[i], 'X': x[i], 'Y': y[i], 'Z': z[i]} for i in range(len(g))]
    starting_point = next(i for i, event in enumerate(event_series) if event['Z'] == layer_height)
    event_series = event_series[starting_point:]

    workpiece_thickness = max(event['Z'] for event in event_series)
    print("Workpiece thickness = " + str(workpiece_thickness) + " mm.")

    return workpiece_thickness, layer_height, event_series, nozzle_diameter, file

def process_layer_data(workpiece_thickness, layer_height, event_series):
    layer_data = []
    num_loops = int(workpiece_thickness / layer_height)

    for loop in range(num_loops):
        print("Iteration " + str(loop + 1))
        desired_z = round((loop + 1) * layer_height, 2)
        current_layer_data = [event for event in event_series if event['Z'] == desired_z]
        layer_data.append(current_layer_data)

        for event in current_layer_data[:5]:
            print(event)
        print("Z level = " + str(desired_z))

    return layer_data

def shift_data(layer_data):
    x_values = [point['X'] for iteration in layer_data for point in iteration]
    y_values = [point['Y'] for iteration in layer_data for point in iteration]

    initial_x_center = (np.min(x_values) + np.max(x_values)) / 2
    initial_y_center = (np.min(y_values) + np.max(y_values)) / 2

    print("Initial X center (midpoint): " + str(initial_x_center))
    print("Initial Y center (midpoint): " + str(initial_y_center))

    for iteration in layer_data:
        for point in iteration:
            point['X'] -= initial_x_center
            point['Y'] -= initial_y_center

    for i, iteration in enumerate(layer_data):
        print("Iteration " + str(i + 1) + ":")
        for point in iteration:
            print(point)

    return point, layer_data

def arrange_data(layer_data):
    arranged_data = []
    for layer in layer_data:
        if len(layer) < 2:
            continue
        for i in range(len(layer) - 1):
            point_pair = (layer[i], layer[i + 1])
            arranged_data.append(point_pair)
    return arranged_data


def generate_filament_model(layer_data, layer_height, nozzle_diameter):
    # Create a new assembly
    my_assembly = mdb.models['Model-1'].rootAssembly
    mdb.models['Model-1'].Material(name='Material-1')
    mdb.models['Model-1'].materials['Material-1'].Elastic(table=((3250.0, 0.3), ))

    mdb.models['Model-1'].materials['Material-1'].Plastic(
        table=((48.75, 0.0), (52.5, 0.019), (45.5, 0.063))
        )

    mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', material='Material-1', thickness=None)


    # Loop over all layers
    for layer_index in range(len(layer_data)):
        layer = layer_data[layer_index]  # Get the current layer data
        
        
        # Calculate the Z position for the current layer
        z_position = (layer_index) * layer_height  # Actual Z value based on layer index
        
        # Print the current layer index for tracking
        print("Creating parts for Layer " + str(layer_index + 1))

        # List to store all instances in the current layer
        layer_instances = []
        parts_to_delete = []  # List to track parts to delete later00

        # Create parts for each pair within the layer
        for i in range(len(layer) - 1):
            point1 = layer[i]
            point2 = layer[i + 1]
            x0, y0 = point1['X'], point1['Y']
            x1, y1 = point2['X'], point2['Y']

            # Extract G values for the current pair
            g1, g2 = point1['G'], point2['G']

            # Check the conditions for including the pair
            if (g1 == 0 and g2 == 1) or (g1 == 1 and g2 == 1):
                # Calculate the length of the rectangle based on the distance
                length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)  # Length based on distance

                # Create a new part for this pair
                part_name = 'Part_Layer_' + str(layer_index + 1) + '_Pair_' + str(i + 1)
                my_model = mdb.models['Model-1']  # Change this to your actual model name
                sketch = my_model.ConstrainedSketch(name=part_name + '_Sketch', sheetSize=200.0)

                # Draw the rectangle centered at (0, 0)
                sketch.rectangle(point1=(0, nozzle_diameter / 2), 
                                 point2=(length, -nozzle_diameter / 2))

                # Create the part from the sketch with adjustable thickness
                my_part = my_model.Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
                my_part.BaseSolidExtrude(sketch=sketch, depth=layer_height)  # Use layer_height for extrusion depth

                # Clean up the sketch
                del my_model.sketches[sketch.name]

                # Add the part to the assembly
                instance_name = part_name + '_Instance'
                my_assembly.Instance(name=instance_name, part=my_part, dependent=ON)

                # Translate the part to the desired coordinates (x0, y0, z_position)
                my_assembly.translate(instanceList=(instance_name,), 
                                       vector=(x0, y0, z_position))

                # Calculate the rotation angle using np.arctan2 for rotation around the Z-axis
                rotation_angle = np.arctan2(y1 - y0, x1 - x0)  # Angle in radians

                # Rotate the part around the Z-axis by the specified angle
                my_assembly.rotate(instanceList=(instance_name,), 
                                   angle=rotation_angle * (180/np.pi),  # Convert radians to degrees
                                   axisPoint=(x0, y0, z_position), 
                                   axisDirection=(0.0, 0.0, 1.0))  # Rotate around Z-axis

                # Print confirmation of part creation, translation, and rotation
                print("Created, translated, and rotated " + part_name + " to position ({}, {}, {})".format(x0, y0, z_position))

                # Add the instance name to the list for merging later
                layer_instances.append(my_assembly.instances[instance_name])

                # Track the part to delete later
                parts_to_delete.append(my_part)

                # Adding a cylinder at the start of the part
                cylinder_name_start = 'Start_Cylinder_Layer_' + str(layer_index + 1) + '_Pair_' + str(i + 1)
                start_sketch = my_model.ConstrainedSketch(name=cylinder_name_start + '_Sketch', sheetSize=200.0)

                # Draw the circle for the base of the start cylinder
                start_sketch.CircleByCenterPerimeter(center=(0, 0), point1=(nozzle_diameter / 2, 0))

                # Create the cylinder part for the start
                start_cylinder_part = my_model.Part(name=cylinder_name_start, dimensionality=THREE_D, type=DEFORMABLE_BODY)
                start_cylinder_part.BaseSolidExtrude(sketch=start_sketch, depth=layer_height)

                # Clean up the start cylinder sketch
                del my_model.sketches[start_sketch.name]

                # Add the start cylinder to the assembly
                start_instance_name = cylinder_name_start + '_Instance'
                my_assembly.Instance(name=start_instance_name, part=start_cylinder_part, dependent=ON)

                # Translate the start cylinder to the start point of the part
                my_assembly.translate(instanceList=(start_instance_name,), 
                                      vector=(x0, y0, z_position))

                #Rotate the start cylinder to align with the part
                my_assembly.rotate(instanceList=(start_instance_name,), 
                                   angle=rotation_angle * (180/np.pi), 
                                   axisPoint=(x0, y0, z_position), 
                                   axisDirection=(0.0, 0.0, 1.0))

                # Print confirmation of start cylinder creation
                print("Added start cylinder at the beginning of part " + part_name)

                # Add the start cylinder instance name to the list for merging
                layer_instances.append(my_assembly.instances[start_instance_name])

                # Track the start cylinder part to delete later
                parts_to_delete.append(start_cylinder_part)

                # Adding a cylinder at the end of the part
                cylinder_name_end = 'End_Cylinder_Layer_' + str(layer_index + 1) + '_Pair_' + str(i + 1)
                end_sketch = my_model.ConstrainedSketch(name=cylinder_name_end + '_Sketch', sheetSize=200.0)

                # Draw the circle for the base of the end cylinder
                end_sketch.CircleByCenterPerimeter(center=(0, 0), point1=(nozzle_diameter / 2, 0))

                # Create the cylinder part for the end
                end_cylinder_part = my_model.Part(name=cylinder_name_end, dimensionality=THREE_D, type=DEFORMABLE_BODY)
                end_cylinder_part.BaseSolidExtrude(sketch=end_sketch, depth=layer_height)

                # Clean up the end cylinder sketch
                del my_model.sketches[end_sketch.name]

                # Add the end cylinder to the assembly
                end_instance_name = cylinder_name_end + '_Instance'
                my_assembly.Instance(name=end_instance_name, part=end_cylinder_part, dependent=ON)

                # Translate the end cylinder to the end point of the part
                my_assembly.translate(instanceList=(end_instance_name,), 
                                      vector=(x1, y1, z_position))

                # Rotate the end cylinder to align with the part
                my_assembly.rotate(instanceList=(end_instance_name,), 
                                   angle=rotation_angle * (180/np.pi), 
                                   axisPoint=(x1, y1, z_position), 
                                   axisDirection=(0.0, 0.0, 1.0))

                # Print confirmation of end cylinder creation
                print("Added end cylinder at the end of part " + part_name)

                # Add the end cylinder instance name to the list for merging
                layer_instances.append(my_assembly.instances[end_instance_name])

                # Track the end cylinder part to delete later
                parts_to_delete.append(end_cylinder_part)

        # Once all parts and cylinders are created for the current layer, merge them into a single part
        merged_part_name = 'Merged_Part_Layer_' + str(layer_index + 1)
        my_assembly.InstanceFromBooleanMerge(name=merged_part_name, 
                                             instances=tuple(layer_instances), 
                                             originalInstances=DELETE, 
                                             domain=GEOMETRY)
        
        # Print confirmation of the merge
        print("Merged all parts and cylinders into " + merged_part_name + " for Layer " + str(layer_index + 1))
        

        # Delete all unmerged parts after the merge
        for part in parts_to_delete:
            del my_model.parts[part.name]

        # Print confirmation of part deletion
        print("Deleted all individual parts and cylinders for Layer " + str(layer_index + 1))


def find_surf_top_btm(layer_data, layer_height):
    model = mdb.models['Model-1']
    assembly = model.rootAssembly

    tolerance = 0.005

    for i in range(len(layer_data)):
        instance_name = "Merged_Part_Layer_" + str(i + 1) + "-1"

        if instance_name not in assembly.instances:
            print("Instance not found: " + instance_name)
            continue

        instance = assembly.instances[instance_name]
        faces = instance.faces

        z_top = layer_height * (i + 1)
        z_bottom = layer_height * i

        faces_top = faces.getByBoundingBox(zMin=z_top - tolerance, zMax=z_top + tolerance)
        faces_bottom = faces.getByBoundingBox(zMin=z_bottom - tolerance, zMax=z_bottom + tolerance)

        print("Layer " + str(i + 1) + ":")
        print(" - Top face count: " + str(len(faces_top)))
        print(" - Bottom face count: " + str(len(faces_bottom)))

        if len(faces_top) > 0:
            try:
                assembly.Surface(side1Faces=faces_top, name="Surf-top" + str(i + 1))
                print(" - Surf-top" + str(i + 1) + " created.")
            except Exception as e:
                print(" - Failed to create Surf-top" + str(i + 1) + ": " + str(e))
        else:
            print(" - No top faces found for " + instance_name)

        if len(faces_bottom) > 0:
            try:
                assembly.Surface(side1Faces=faces_bottom, name="Surf-btm" + str(i + 1))
                print(" - Surf-btm" + str(i + 1) + " created.")
            except Exception as e:
                print(" - Failed to create Surf-btm" + str(i + 1) + ": " + str(e))
        else:
            print(" - No bottom faces found for " + instance_name)
            
def find_surf_maxmin_x(layer_data):
    # Access the model
    model = mdb.models['Model-1']
    # Access the root assembly
    assembly = model.rootAssembly

    for i in range(len(layer_data)):
        # Define the instance name for the current layer, including '-1' suffix
        instance_name = "Merged_Part_Layer_" + str(i + 1) + "-1"

        # Access the instance in the assembly
        if instance_name in assembly.instances:
            instance = assembly.instances[instance_name]
            faces = instance.faces  # Access the faces of the instance

            # Define the z-coordinates to look for based on the layer index
            target_z_top = 0.2 * (i + 1)  # z for top surface
            target_z_bottom = 0.2 * i      # z for bottom surface
            
            # Initialize empty lists to store the coordinates of faces at the target z and x
            side1FaceCoords_top = []
            side1FaceCoords_bottom = []
            side1FaceCoords_max_x = []
            side1FaceCoords_min_x = []

            # Initialize variables to track the maximum and minimum x values
            max_x = float('-inf')
            min_x = float('inf')
            
            # Loop over each face to check for both target z-coordinates and x-coordinates
            for face in faces:
                try:
                    # Retrieve the centroid and extract the inner tuple
                    center = face.getCentroid()[0]
                    
                    # Update max_x and min_x based on face centroids
                    if center[0] > max_x:
                        max_x = center[0]
                    if center[0] < min_x:
                        min_x = center[0]
                    
                    # Check if z-coordinate is close to the target_z_top
                    if abs(center[2] - target_z_top) < 1e-4:  # Adjust tolerance as needed
                        side1FaceCoords_top.append((center,))
                        
                    # Check if z-coordinate is close to the target_z_bottom
                    if abs(center[2] - target_z_bottom) < 1e-4:  # Adjust tolerance as needed
                        side1FaceCoords_bottom.append((center,))
                        
                except Exception as e:
                    print("Error processing face in " + instance_name + ": " + str(e))

            # After finding max_x and min_x, loop again to find faces at these coordinates
            for face in faces:
                try:
                    center = face.getCentroid()[0]
                    
                    # Check if x-coordinate is close to max_x
                    if abs(center[0] - max_x) < 1e-4:  # Adjust tolerance as needed
                        side1FaceCoords_max_x.append((center,))
                    
                    # Check if x-coordinate is close to min_x
                    if abs(center[0] - min_x) < 1e-4:  # Adjust tolerance as needed
                        side1FaceCoords_min_x.append((center,))
                        
                except Exception as e:
                    print("Error processing face in " + instance_name + ": " + str(e))

            # Find faces using the collected coordinates for top surface
            side1Faces_top = faces.findAt(*side1FaceCoords_top) if side1FaceCoords_top else []
            if side1Faces_top:
                assembly.Surface(side1Faces=side1Faces_top, name="Surf-top" + str(i + 1))
            else:
                print("No faces found at z=" + str(target_z_top) + " for " + instance_name + ".")
            
            # Find faces using the collected coordinates for bottom surface
            side1Faces_bottom = faces.findAt(*side1FaceCoords_bottom) if side1FaceCoords_bottom else []
            if side1Faces_bottom:
                assembly.Surface(side1Faces=side1Faces_bottom, name="Surf-btm" + str(i + 1))
            else:
                print("No faces found at z=" + str(target_z_bottom) + " for " + instance_name + ".")

            # Find faces using the collected coordinates for max x surface
            side1Faces_max_x = faces.findAt(*side1FaceCoords_max_x) if side1FaceCoords_max_x else []
            if side1Faces_max_x:
                assembly.Surface(side1Faces=side1Faces_max_x, name="Surf-maxX" + str(i + 1))
            else:
                print("No faces found at x=" + str(max_x) + " for " + instance_name + ".")
            side1Faces_min_x = faces.findAt(*side1FaceCoords_min_x) if side1FaceCoords_min_x else []
            if side1Faces_min_x:
                assembly.Set(faces=side1Faces_min_x, name="Surf-minX" + str(i + 1))  # Remains as 'Surf-minX'

            else:
                print("No faces found at x=" + str(min_x) + " for " + instance_name + ".")    
                

def tie_interaction(layer_data):
    model = mdb.models['Model-1']
    assembly = model.rootAssembly

    for i in range(len(layer_data) - 1):
        master_surface_name = "Surf-top" + str(i + 1)
        slave_surface_name = "Surf-btm" + str(i + 2)
        constraint_name = "Tie-" + str(i + 1)

        if master_surface_name in assembly.surfaces and slave_surface_name in assembly.surfaces:
            master_surface = assembly.surfaces[master_surface_name]
            slave_surface = assembly.surfaces[slave_surface_name]

            master_faces = master_surface.faces
            slave_faces = slave_surface.faces

            print("Checking tie '" + constraint_name + "': Master " + str(len(master_faces)) + " faces, Slave " + str(len(slave_faces)) + " faces")

            if len(master_faces) < 2:
                instance_name = "Merged_Part_Layer_" + str(i + 1) + "-1"
                if instance_name in assembly.instances:
                    try:
                        assembly.instances[instance_name].PartitionCellByVirtualTopology(cells=assembly.instances[instance_name].cells[:])
                        print("Applied virtual topology to " + instance_name)
                    except Exception as e:
                        print("Virtual topology failed for " + instance_name + ": " + str(e))

            if len(slave_faces) < 2:
                instance_name = "Merged_Part_Layer_" + str(i + 2) + "-1"
                if instance_name in assembly.instances:
                    try:
                        assembly.instances[instance_name].PartitionCellByVirtualTopology(cells=assembly.instances[instance_name].cells[:])
                        print("Applied virtual topology to " + instance_name)
                    except Exception as e:
                        print("Virtual topology failed for " + instance_name + ": " + str(e))

            master_surface = assembly.surfaces[master_surface_name]
            slave_surface = assembly.surfaces[slave_surface_name]
            master_faces = master_surface.faces
            slave_faces = slave_surface.faces

            if len(master_faces) >= 1 and len(slave_faces) >= 1:
                try:
                    model.Tie(name=constraint_name, master=master_surface, slave=slave_surface,
                              positionToleranceMethod=COMPUTED, adjust=ON,
                              tieRotations=ON, thickness=ON)
                    print("Tie '" + constraint_name + "' created successfully.")
                except Exception as e:
                    print("Final Tie '" + constraint_name + "' failed: " + str(e))
            else:
                print("Skipped tie '" + constraint_name + "' due to empty face list even after virtual topology.")
        else:
            print("Surface missing: " + master_surface_name + " or " + slave_surface_name)


def assign_section(layer_data):
    #Assign the section
    for i in range(len(layer_data)):
        # Assign section properties to the entire part
        part_name = 'Merged_Part_Layer_' + str(i + 1)
        part = mdb.models['Model-1'].parts[part_name]
        cells = part.cells
        region = (cells,)
        part.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
###############################################################################
def assign_step():
    # Generate a step for the simulation
    step_name = 'Step-1'
    mdb.models['Model-1'].StaticStep(name=step_name, previous='Initial', description='Static analysis step for the 3D part').setValues(initialInc=0.1)
###############################################################################
def assign_RF_point(layer_data):
    # Access the model
    model = mdb.models['Model-1']
    # Access the root assembly
    assembly = model.rootAssembly
    # Flatten the list of dictionaries to access values directly
    x_values = [point['X'] for layer in layer_data for point in layer]
    y_values = [point['Y'] for layer in layer_data for point in layer]
    z_values = [point['Z'] for layer in layer_data for point in layer]

    # Calculate max and min for X
    max_x = max(x_values)
    min_x = min(x_values)

    # Calculate max and min for Y and Z
    max_y = max(y_values)
    min_y = min(y_values)
    max_z = max(z_values)
    min_z = min(z_values)

    # Calculate middle position for Y and Z
    middle_y = (max_y + min_y) / 2
    middle_z = (max_z + 0) / 2

    # Output the results using traditional formatting
    print("Max X: {}".format(max_x))
    print("Min X: {}".format(min_x))
    print("Max Y: {}".format(max_y))
    print("Min Y: {}".format(min_y))
    print("Middle Y: {}".format(middle_y))
    print("Middle Z: {}".format(middle_z))

    # Create the reference point using calculated coordinates
    # Create a reference point
    point = (max_x + 2, middle_y, middle_z)
    # Create the reference point in the assembly
    referencePoint = assembly.ReferencePoint(point=point)

    # Create a set containing the reference point using its unique ID
    rf_set_name = 'RF_Set'
    rf_set = assembly.Set(name=rf_set_name, referencePoints=(assembly.referencePoints[referencePoint.id],))
    return rf_set
###############################################################################
def assign_coupling_interaction(layer_data, rf_set):
    # Access the model
    model = mdb.models['Model-1']
    # Access the root assembly
    assembly = model.rootAssembly
    # Loop over each layer to create coupling constraints for the reference point
    for i in range(len(layer_data)-2):
        # Define the name for the surface created in the previous sections
        surface_name = "Surf-maxX" + str(i + 1)
        
        # Access the model
        model = mdb.models['Model-1']
        assembly = model.rootAssembly

        # Create a region for the coupling using the rf_set
        ref_point_region = rf_set

        # Create the coupling constraint
        model.Coupling(name='Coupling-RP-' + str(i + 1),
                       controlPoint=ref_point_region,
                       surface=assembly.surfaces[surface_name],  # Accessing the surface directly
                       influenceRadius=WHOLE_SURFACE,
                       couplingType=KINEMATIC)
###############################################################################
def assign_boundary_condition(layer_data, rf_set):
    #Apply fix boundary condition
    for i in range(len(layer_data)-2):
        # Define the name for the surface
        surface_name = "Surf-minX" + str(i + 1)
        
        # Access the model
        model = mdb.models['Model-1']
        assembly = model.rootAssembly

        model.EncastreBC(name='BC-MinX-' + str(i + 1), 
                     createStepName='Initial', 
                     region=assembly.sets['Surf-minX' + str(i + 1)])
    # Get user input for displacement value in X direction
    rotation_around_x = float(getInputs(fields=[('Rotation around X axis (Rad)', '0.0')])[0])   
    #Apply displacement boundary condition
    model.DisplacementBC(name='BC-RefPoint', createStepName='Initial', region=rf_set, 
                         u1=0, u2=0, u3=0, ur1=0, ur2=0, ur3=0)    
    model.DisplacementBC(name='BC-RefPoint', createStepName='Step-1', region=rf_set, 
                         u1=0, u2=0, u3=0, ur1=rotation_around_x, ur2=0, ur3=0)    
###############################################################################
def assign_mesh():
    # Ask user for the desired mesh size
    mesh_size_input = getInputs(fields=[('Enter desired mesh size', '')])[0].strip()

    # Convert input to float
    mesh_size = float(mesh_size_input)

    # Define the element type for wedge elements (C3D6 or C3D6R)
    elemType1 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)

    # Automatically mesh each part using the user-defined mesh size
    for part_name in mdb.models['Model-1'].parts.keys():
        part = mdb.models['Model-1'].parts[part_name]

        # Assign element type to all cells
        part.setElementType(regions=(part.cells,), elemTypes=(elemType1,))

        # Set mesh controls to use wedge elements and sweep technique
        part.setMeshControls(regions=part.cells, elemShape=WEDGE, technique=SWEEP)

        # Seed the part with the specified mesh size
        part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)

        # Generate the mesh
        part.generateMesh()

        # Print confirmation
        print('Meshed part: ' + str(part_name) + ' with WEDGE elements and mesh size ' + str(mesh_size))


###############################################################################
def create_job(file):
    # Create and submit a job
    job_name = file
    mdb.Job(name=job_name, model='Model-1', description='Simulation job based on G-code.')
###############################################################################
def main():
    workpiece_thickness, layer_height, event_series, nozzle_diameter, file = read_gcode()
    layer_data = process_layer_data(workpiece_thickness, layer_height, event_series)
    point, layer_data = shift_data(layer_data)
    #arranged_data = arrange_data(layer_data)
    generate_filament_model(layer_data, layer_height, nozzle_diameter)    
    find_surf_top_btm(layer_data, layer_height)
    find_surf_maxmin_x(layer_data)
    tie_interaction(layer_data)
    assign_section(layer_data)
    assign_step()
    rf_set = assign_RF_point(layer_data)
    assign_mesh()
    assign_boundary_condition(layer_data, rf_set)
    assign_coupling_interaction(layer_data, rf_set)
    create_job(file)

# Execute main function if the script is run directly
if __name__ == "__main__":
    main()