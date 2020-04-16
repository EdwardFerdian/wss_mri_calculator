import pyvista as pv
import numpy as np
import argparse
import logging
import data_loader
import wss_utils

def create_uniform_grid(mask, spacing):
    mesh = pv.UniformGrid()
    mesh.dimensions = np.array(mask.shape) + 1

    # grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
    mesh.spacing = spacing # These are the cell sizes along each axis
    mesh.cell_arrays["mask"] = mask.flatten(order="F") 
    return mesh

def create_uniform_vector(u,v,w, spacing):
    vel = np.sqrt(u **2 + v**2 + w**2)

    mesh = pv.UniformGrid()
    mesh.dimensions = np.array(u.shape) + 1

    # grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
    mesh.spacing = spacing  # These are the cell sizes along each axis
    
    mesh.cell_arrays["u"] = u.flatten(order="F")  # Flatten the array!
    mesh.cell_arrays["v"] = v.flatten(order="F") 
    mesh.cell_arrays["w"] = w.flatten(order="F")

    mesh.cell_arrays["Velocity"] = vel.flatten(order="F")
    mesh.set_active_scalars("Velocity")
    return mesh

def boolean_string(s):
    """
        Python argparse for bool consider the input as str. 
        Handle it with this function
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default='../data/aorta03_sample.h5', help='Phase images or velocity vectors, containing u,v,w vector values')
    parser.add_argument('--mask-file', type=str, default='../data/aorta03_sample.h5', help='Mask/segmentation file. Can also be a .stl with another loader')
    parser.add_argument('--voxel-size', type=float, default=0.6, help='Voxel size, assumed to be isotropic (in mm)')
    parser.add_argument('--inward-distance', type=float, default=0.6, help='Inward normal distance to sample points from wall (in mm)')
    parser.add_argument('--smoothing', type=int, default=500, help='Number of iterations to adjust point coordinates using Laplacian smoothing.')
    parser.add_argument('--parabolic', type=boolean_string, default=True, help='Use parabolic curve fitting to determine the slope. When False use linear.')
    parser.add_argument('--no-slip', type=boolean_string, default=True, help='Set the wall velocity to zero for WSS calculation')
    parser.add_argument('--viscosity', type=float, default=4, help='Fluid viscosity (default is 4centiPoise - blood viscosity)')
    parser.add_argument('--show-plot', type=boolean_string, default=True, help='Plot the images using PyVista plotter')
    parser.add_argument('--show-wss-contours', type=boolean_string, default=False, help='Show WSS contours for visualization')
    parser.add_argument('--save-to-vtk', type=float, default=False, help='Save volume mesh and surface WSS in vtk file')
    parser.add_argument('--vtk-filename', type=str, default='result', help='Prefix for the output file when save_to_vtk is True')
    parser.add_argument('--loglevel', type=int, default=logging.INFO, help='Logging level')
    
    args = parser.parse_args()
    
    print('\nArguments:', args)
    # set the logging level on the imported script
    logging.getLogger("wss_utils").setLevel(args.loglevel)  

    logging.basicConfig()
    logger = logging.getLogger('calculate_wss')
    logger.setLevel(args.loglevel)
    
    # ----------- Parameters ----------------------
    # If the dimension is not isotropic, you might as well edit it here
    spacing = (args.voxel_size,args.voxel_size,args.voxel_size) # in mm
    inward_distance = args.inward_distance
    smoothing_iteration = args.smoothing

    no_slip_condition= args.no_slip
    parabolic_fitting = args.parabolic
    viscosity = args.viscosity # in centiPoise, velocities are in m/s so they cancel out each other

    # Specify input filepath and segmentation file path
    input_filepath = args.input_file
    mask_filepath = args.mask_file

    # 1. Load velocity vectors and segmentation/mask
    # Note: Please replace the loader depending on your data format, this is just an example
    logger.info("Loading velocity vectors")
    u,v,w = data_loader.load_vector_fields(input_filepath, ['u', 'v', 'w'], 0)
    logger.debug("Image shape {}".format(u.shape))

    logger.info("Load segmentation")
    mask = data_loader.load_segmentation(mask_filepath, 'mask', 0)

    # 2. Construct uniform grid for vector values and mask
    logger.info("Constructing uniform grids for vectors and mask")
    velocity = create_uniform_vector(u,v,w, spacing)
    mesh = create_uniform_grid(mask, spacing)
    mesh = mesh.threshold_percent(40)
    # mesh = mesh.threshold([0.3,0.8])
    
    # 3. Get the volume within the mask (for visualization purpose)
    logger.info("Get volume")
    volume = mesh.sample(velocity)
    volume.set_active_scalars("Velocity")

    # 4. Get the surface of the mesh
    # Note: do not use extract_geometry, the PyVista manual on the website is misleading
    logger.info("Extracting surface")
    surf = mesh.extract_surface()
    surf = surf.smooth(n_iter=smoothing_iteration)

    # 5. Calculate the point normals (depends on what we computed on cells or on nodes)
    # Make sure we compute only the point normals, not the cell normals, they have the same array name
    logger.info("Computing surface normals")
    surf.compute_normals(point_normals=True, cell_normals=True, inplace=True, flip_normals=True)
    logger.debug("{} {} normal points".format(surf.point_normals[0:2], len(surf.point_normals)))
    logger.debug("{} {} points coords".format(surf.points[0:2], len(surf.points)))

    # 6. Compute the coordinates from surface (pc0) to the equidistant points in the inward normal (pc1 and pc2)
    # Construct a polydata of points from the surface points
    logger.info("Probing velocity from surface and equidistant points {} nodes".format(len(surf.points)))
    pc0 = pv.PolyData(surf.points)
    pc0 = pc0.sample(velocity)
    pc0.set_active_scalars("Velocity")

    # Construct the first inward point clouds form surface (pc1) and get the velocity values
    loc1 = pc0.points + (inward_distance * surf.point_normals)
    pc1 = pv.PolyData(loc1)
    pc1 = pc1.sample(velocity)

    # Construct the second inward point clouds form surface (pc2) and get the velocity values
    loc2 = pc0.points + (2 * inward_distance * surf.point_normals)
    pc2 = pv.PolyData(loc2)
    pc2 = pc2.sample(velocity)

    # 7. Calculate the tangential vector on the wall and inward points
    logger.info("Calculate normal and tangential velocity vectors")
    if no_slip_condition:
        # For the velocity on the wall to zero
        pc0_tangent_mag = np.zeros(len(pc0.points))
    else:
        # Extract the velocity on the wall. This might not be accurate as it depends on the segmentation
        pc0_vectors = wss_utils.extract_vectors(pc0)
        pc0_normals, pc0_tangent = wss_utils.get_orthogonal_vectors(pc0_vectors, surf.point_normals)
        pc0_tangent_mag = wss_utils.get_vector_magnitude(pc0_tangent)
    logger.debug("Tangent vector pc0 {}".format(pc0_tangent_mag[0:2]))

    # get pc1 vector normal and tangential
    pc1_vectors = wss_utils.extract_vectors(pc1)
    pc1_normals, pc1_tangent = wss_utils.get_orthogonal_vectors(pc1_vectors, surf.point_normals)
    pc1_tangent_mag = wss_utils.get_vector_magnitude(pc1_tangent)
    pc1['vectors'] = pc1_tangent

    # get pc2 vector normal and tangential
    pc2_vectors = wss_utils.extract_vectors(pc2)
    pc2_normals, pc2_tangent = wss_utils.get_orthogonal_vectors(pc2_vectors, surf.point_normals)
    pc2_tangent_mag = wss_utils.get_vector_magnitude(pc2_tangent)
    pc2['vectors'] = pc2_tangent

    # === Experimental ==
    # Check if the tangential vectors of pc1 and pc2 are in the same direction 
    c = pc1_tangent * pc2_tangent 
    c = np.sum(c, axis=1)
    c = c.clip(min = -1, max = 1) # Assign the direction
    pc2_tangent_mag =  c * pc2_tangent_mag # Give negative magnitude value if pc2 is in different direction
    # We don't check based on pc0 because we cannot trust the direction of pc0 vectors which may be affected by segmentation
    ## == end ==

    logger.debug("Tangent vector pc1 {}".format(pc1_tangent_mag[0:2]))
    logger.debug("Tangent vector pc2 {}".format(pc2_tangent_mag[0:2]))

    # 8. Compute the gradients for every points
    logger.info("Spline fitting and calculating gradients ...")
    gradients = wss_utils.calculate_gradient(pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic=parabolic_fitting)
    logger.debug("gradients {}".format(gradients[0:2]))

    # 9. Assign the gradients back to the surface
    logger.info("Calculating WSS")
    surf["wss"] = gradients * args.viscosity
    # use the pc1 vectors for visualization
    surf['wss_vectors'] = pc1_tangent

    # 10. Save the whole volume and the surface
    # TODO: Improve this so we can save volume and surface in 1 vtk file
    if args.save_to_vtk:
        logger.info("Saving to {}_surface.vtk".format(args.vtk_filename))
        volume.save("{}_volume.vtk".format(args.vtk_filename))
        surf.save("{}_surface.vtk".format(args.vtk_filename))

    if args.show_plot:
        # ============ Plot the images ============ 
        logger.info("Preparing plot...")
        p = pv.Plotter(notebook=0, shape=(2, 2), border=False)
        
        # === Column 1
        p.subplot(0, 0)
        p.add_text("Voxels\n{} mm".format(args.voxel_size), font_size=20)
        p.add_mesh(volume, show_edges=True, cmap='jet')

        # === Column 2
        p.subplot(0, 1)
        p.add_text("Surface (n={})".format(smoothing_iteration), font_size=20)
        result = surf.sample(velocity)
        p.add_mesh(result, scalars="Velocity", opacity=0.7, cmap='viridis')
        # Add some arrows just because :)
        arrows = surf.glyph(scale="Normals", orient="Normals", tolerance=0.05)
        p.add_mesh(arrows, color="black")
        p.show_bounds(all_edges=True)

        # === Row 2
        p.subplot(1, 0)
        p.add_text("Wall points\n(inward {})".format(inward_distance), font_size=20)
        # Add the points only, for visualization purpose
        p.add_mesh(pc0, cmap='jet', opacity=0.5)
        # Add glyphs for pc1 and pc2
        pc1_arrows = pc1.glyph(orient='vectors', scale=False, factor=0.6,)
        pc2_arrows = pc2.glyph(orient='vectors', scale=False, factor=0.6,)
        p.add_mesh(pc1_arrows, color='black')
        p.add_mesh(pc2_arrows, color='red')

        # === Row 2, col 2
        p.subplot(1, 1)
        fitting_functext = "parabolic" if parabolic_fitting else "linear"
        p.add_text("WSS ({})".format(fitting_functext), font_size=20)
        p.add_mesh(surf, scalars="wss", cmap='jet')

        if args.show_wss_contours:
            contours = surf.contour()
            p.add_mesh(contours, color='black', line_width=1)

        # ===================================
        # Link all the views and show the plot
        p.link_views()
        p.show(full_screen=True)

if __name__ == "__main__":
    main()