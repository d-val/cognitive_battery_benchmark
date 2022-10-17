# ## Initializing Objects Point Clouds

# +
# import CoraAgent as CA
import InverseGraphics as T
import Open3DVisualizer as V
import MeshCatViz as MV
import Gen

import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import LightGraphs as LG
import ImageView as IV
import PyCall
np = PyCall.pyimport("numpy")
MV.setup_visualizer()
# -

function CameraIntrinsics(width, height, fov_y_deg)
    aspect_ratio = width / height

    # Camera principal point is the center of the image.
    cx, cy = width / 2.0, height / 2.0

    # Vertical field of view is given.
    fov_y = deg2rad(fov_y_deg)
    # Convert field of view to distance to scale by aspect ratio and
    # convert back to radians to recover the horizontal field of view.
    fov_x = 2 * atan(aspect_ratio * tan(fov_y / 2.0))

    # Use the following relation to recover the focal length:
    #   FOV = 2 * atan( (0.5 * IMAGE_PLANE_SIZE) / FOCAL_LENGTH )
    fx = cx / tan(fov_x / 2.0)
    fy = cy / tan(fov_y / 2.0)

    clipping_near, clipping_far = 0.01, 100000.0

    GL.CameraIntrinsics(width, height,
        fx, fy, cx, cy,
        clipping_near, clipping_far)
end

function find_table_plane(cloud; max_iters=5)
    pyrsc = PyCall.pyimport("pyransac3d")
    for it in 1:max_iters
        plane1 = pyrsc.Plane()
        best_eq, _ = plane1.fit(transpose(cloud), 0.1)

        if 0.4 < abs(best_eq[2]) < 0.5
            print("Iteration ");
            print(it);
            print(" ")
            print(best_eq[2])
            print("\n")
            return best_eq
        else
            inliers = abs.((best_eq[1:3]' * cloud)[:] .+ best_eq[4]) .< 0.1
            mask = fill(false, size(cloud)[2])
            mask[inliers] .= true
            cloud = cloud[:, .!(mask)]
        end
    end
    return nothing
end

function get_entities_from_scene(frame_path, depth_path, intrinsics)
    frame_image = T.load_rgb(frame_path)
    depth_image = np.load(depth_path)
    
    point_cloud = GL.depth_image_to_point_cloud(depth_image, intrinsics)

    cropped_point_cloud = point_cloud[:, [i for i in 1: size(point_cloud)[2] 
                if (
                        -.5 < point_cloud[1,i] < 1
                    &&  -.5 < point_cloud[2,i] < 1
                    &&  1.2 < point_cloud[3,i] < 4
                )]]
    
    # Getting table description
    table_eq = find_table_plane(cropped_point_cloud, max_iters=1000)
    
    # Mask: which columns correspond to the y position of the objects
    # Inliers: the objects (point cloud of the object)
    inliers, mask = T.find_plane_inliers(cropped_point_cloud, table_eq; threshold = 0.1); 

    # This takes objects that are not in the table (mask defines where the table is)
    just_objects_point_cloud =  cropped_point_cloud[:,[i for i in 1:length(mask) if mask[i]==0]]

    # Extracting entities from objects point cloud
    entities = T.get_entities_from_assignment(just_objects_point_cloud, T.dbscan_cluster(just_objects_point_cloud; radius=0.1505))
    
    return entities
end

data_path = dirname(joinpath(
#     dirname(dirname(pathof(T))),
    "data/"
))
frames_paths = [
    joinpath(data_path, "frames/frame_$i.jpeg")
    for i in 1:100
]
depths_paths = [
    joinpath(data_path, "depths/frame_$i.npy")
    for i in 1:100
];
segmented_paths = [
    joinpath(data_path, "segmented/frame_$i.npy")
    for i in 1:100
];

# Get camera intrinsics from first image
depth_image = np.load(depths_paths[1])
intrinsics = CameraIntrinsics(
        size(depth_image, 2),
        size(depth_image, 1),
        90
    )

# +
frame_image = T.load_rgb(frames_paths[1])
depth_image = np.load(depths_paths[1])
segmented_image = np.load(segmented_paths[1], )

point_cloud = GL.depth_image_to_point_cloud(depth_image, intrinsics)
GL.point_cloud_to_pixel_coordinates(point_cloud, intrinsics)
# -

# Get entities over time
num_frames = length(frames_paths)
num_frames = 1
entities_over_time = [
    get_entities_from_scene(frames_paths[i],depths_paths[i], intrinsics)   
    for i = 53:55
];

MV.reset_visualizer()
first_frame = entities_over_time[1]
MV.viz(hcat(first_frame...)/10)

# +
depth_image = np.load(depths_paths[53])

point_cloud = GL.depth_image_to_point_cloud(depth_image, intrinsics)

cropped_point_cloud = point_cloud[:, [i for i in 1: size(point_cloud)[2] 
            if (
                -.5 < point_cloud[1,i] < 1
            &&  -5 < point_cloud[2,i] < 1 # y
            &&  1.2 < point_cloud[3,i] < 4
            )]]

MV.viz(cropped_point_cloud/10)
# -

entities_over_time[1]

# ## After Multiple Timestep Inference

Gen.@gen function multiobject_tracking_init_state(num_timesteps, renderer, num_shapes, v_resolution)
    N ~ Gen.poisson(5)
    init_shapes = Int[]
    
    for i=1:N
        shape = {i => :shape} ~ Gen.categorical(ones(num_shapes) ./ num_shapes)
        push!(init_shapes, shape)
    end

    poses_at_each_timestep = []
    voxelized_point_clouds = []
    observations = []
    for t in 1:num_timesteps
        poses = T.Pose[] 
        if t==1
            for i=1:N
                pose = {t => i => :pose} ~ T.uniformPose(-1000.0,1000.0,-1000.0,1000.0,-1000.0,1000.0)
                push!(poses, pose)
            end
        else
            for i=1:N
                pose = {t => i => :pose} ~ T.gaussianVMF(poses_at_each_timestep[end][i], 0.05, 2000.0)
                push!(poses, pose)
            end
        end
        push!(poses_at_each_timestep, poses)

        rendered_depth_image = T.GL.gl_render(
            renderer,
            init_shapes,
            poses,
            T.IDENTITY_POSE # camera pose
        )
        point_cloud = T.GL.depth_image_to_point_cloud(rendered_depth_image, renderer.camera_intrinsics)
        voxelized_point_cloud = T.voxelize(point_cloud, v_resolution)
        obs = {t => :obs} ~ T.volume_cloud_likelihood(
            voxelized_point_cloud,
            v_resolution,
            0.1,
            (-100.0,100.0,-100.0,100.0,-100.0,300.0)
        )

        push!(voxelized_point_clouds, voxelized_point_cloud)
        push!(observations, obs)
    end
    (;voxelized_point_clouds, observations, poses_at_each_timestep)
end

t = 6
es = entities_over_time[t]

shape_models = []
initial_poses = []
multiplier = 10.0
for c in es
    c = multiplier .* c
    centroid = T.centroid(c)
    push!(shape_models, c .- centroid)
    push!(initial_poses, T.Pose(centroid, T.IDENTITY_ORN))
end

resolution = 0.25
scaled_down_intrinsics = T.GL.scale_down_camera(intrinsics, 4)
renderer = T.GL.setup_renderer(scaled_down_intrinsics, T.GL.DepthMode(); gl_version=(3,3))
for s in shape_models
    m = T.GL.mesh_from_voxelized_cloud(s, resolution)
    T.GL.load_object!(renderer, m)
end

rendered_depth_image = T.GL.gl_render(
    renderer,
    Vector{Int}(1:length(initial_poses)),
    Vector{T.Pose}(initial_poses),
    T.IDENTITY_POSE # camera pose
)
T.FileIO.save("rerendered_depth.png", T.GL.view_depth_image(rendered_depth_image));

constraints = Gen.choicemap()
constraints[:N] = length(shape_models)
for i=1:length(shape_models)
    constraints[i => :shape] = i
    constraints[1 => i => :pose] = initial_poses[i]
end
constraints[1 => :obs] = T.voxelize(hcat(entities_over_time[t]...) .* multiplier, resolution)

@time trace, _ = Gen.generate(multiobject_tracking_init_state, (1, renderer, length(shape_models), resolution), constraints);
MV.reset_visualizer()
MV.viz(Gen.get_retval(trace).voxelized_point_clouds[1] ./500.0; color=T.I.colorant"black", channel_name=:gen_cloud)
# MV.viz(Gen.get_retval(trace).observations[1] ./500.0; color=T.I.colorant"red", channel_name=:obs_cloud)

new_args = (2, renderer, length(shape_models), resolution)
new_arg_diff = Tuple([Gen.UnknownChange() for _=1:length(new_args)]);

constraints[2 => :obs] = T.voxelize(hcat(entities_over_time[t+1]...) .* multiplier, resolution)
@time trace, _ = Gen.update(trace, new_args, new_arg_diff, constraints);

MV.reset_visualizer()
MV.viz(Gen.get_retval(trace).voxelized_point_clouds[2] ./500.0; color=T.I.colorant"black", channel_name=:gen_cloud)
MV.viz(Gen.get_retval(trace).observations[2] ./500.0; color=T.I.colorant"red", channel_name=:obs_cloud)

for object_id in 1:3
    for _ in 1:100
        trace, acc = T.drift_move(trace, 2 => object_id => :pose, 0.05, 1000.0);
    end
end

MV.reset_visualizer()
MV.viz(Gen.get_retval(trace).voxelized_point_clouds[2] ./500.0; color=T.I.colorant"black", channel_name=:gen_cloud)
MV.viz(Gen.get_retval(trace).observations[2] ./500.0; color=T.I.colorant"red", channel_name=:obs_cloud)
