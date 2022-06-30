import CoraAgent as CA
import InverseGraphics as T
import Open3DVisualizer as V
import MeshCatViz as MV
import Gen

agent_state = CA.setup_mcs("level2")

CA.load_scene!(agent_state, joinpath(CA.get_cora_agent_module_path(),"assets/scene_jsons/intphys.json"))
for _ in 1:100
    CA.execute_command!(agent_state, "Pass")
    CA.update_pose_by_integrating!(agent_state)
end

V.open_window(agent_state.intrinsics, agent_state.pose_history[1]);


clouds = [CA.get_point_cloud_augmented_with_seg_ids(agent_state,t) for t in 1:length(agent_state.step_metadata_history)];
resolution = 0.02
entities_over_time = [
    T.GL.voxelize.(CA.get_entities_from_point_cloud_augmented_with_seg_ids(
        c[:, (c[2,:] .< 1.49) .& (c[3,:] .< 9.49)]
    ), resolution
    )
    for c in clouds
];

# colors = T.I.distinguishable_colors(10, cchoices=[30,40,50,60,70,80,90,100],lchoices=[80]);

# for (t,es) in enumerate(entities_over_time)
#     V.clear()
#     for (index, i) in enumerate(es)
#         V.make_point_cloud(i; color=colors[index])
#     end
#     img = V.capture_image()
#     T.FileIO.save(T.temp_file_name(t), T.GL.view_rgb_image(img))
# end
# gif = T.gif_from_filenames(T.temp_file_name.(1:length(entities_over_time)));
# T.FileIO.save("temp.gif",gif);



# i = 5
# V.clear()
# V.make_point_cloud(T.move_points_to_frame_b(initial_shapes[i], initial_poses[i]); color=colors[i])


# V.destroy()

Gen.@gen function multiobject_tracking_init_state(renderer, num_shapes, v_resolution)
    N ~ Gen.poisson(5)
    init_shapes = Int[]
    init_poses = T.Pose[]
    

    for i=1:N
        shape = {i => :shape} ~ Gen.categorical(ones(num_shapes) ./ num_shapes)
        pose = {i => :pose} ~ T.uniformPose(-1000.0,1000.0,-1000.0,1000.0,-1000.0,1000.0)
        push!(init_shapes, shape)
        push!(init_poses, pose)
    end

    rendered_depth_image = T.GL.gl_render(
        renderer,
        init_shapes,
        init_poses,
        T.IDENTITY_POSE # camera pose
    )
    point_cloud = T.GL.depth_image_to_point_cloud(rendered_depth_image, renderer.camera_intrinsics)
    voxelized_point_cloud = T.voxelize(point_cloud, v_resolution)
    obs = {T.obs_addr()} ~ T.volume_cloud_likelihood(
        voxelized_point_cloud,
        v_resolution,
        0.1,
        (-100.0,100.0,-100.0,100.0,-100.0,300.0)
    )
    (;rendered_depth_image, point_cloud, obs)
end

t = 96 
es = entities_over_time[t]

shape_models = []
initial_poses = []
for c in es
    centroid = T.centroid(c)
    push!(shape_models, c .- centroid)
    push!(initial_poses, T.Pose(centroid, T.IDENTITY_ORN))
end

scaled_down_intrinsics = T.GL.scale_down_camera(agent_state.intrinsics, 4)
renderer = T.GL.setup_renderer(scaled_down_intrinsics, T.GL.DepthMode(); gl_version=(4,1))
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
    constraints[i => :pose] = initial_poses[i]
end
constraints[T.obs_addr()] = T.voxelize(hcat(entities_over_time[t]...), resolution)

@time trace, _ = Gen.generate(multiobject_tracking_init_state, (renderer, length(shape_models), resolution), constraints);
d = Gen.get_retval(trace).rendered_depth_image;
hypothesized_point_cloud = Gen.get_retval(trace).point_cloud
real_obs_point_cloud = Gen.get_retval(trace).obs

MV.setup_visualizer()
MV.viz(hypothesized_point_cloud ./10.0; color=T.I.colorant"black", channel_name=:gen_cloud)
MV.viz(real_obs_point_cloud ./10.0; color=T.I.colorant"red", channel_name=:obs_cloud)

constraints[T.obs_addr()] = T.voxelize(hcat(entities_over_time[t+1]...), resolution)
@time trace, _ = Gen.update(trace, constraints);

hypothesized_point_cloud = Gen.get_retval(trace).point_cloud
real_obs_point_cloud = Gen.get_retval(trace).obs
MV.reset_visualizer()
MV.viz(hypothesized_point_cloud ./10.0; color=T.I.colorant"black", channel_name=:gen_cloud)
MV.viz(real_obs_point_cloud ./10.0; color=T.I.colorant"red", channel_name=:obs_cloud)

MV.reset_visualizer()
MV.viz(shape_models[3] ./10.0; color=T.I.colorant"black", channel_name=:gen_cloud)

for _ in 1:100
    trace, acc = T.drift_move(trace, 3 => :pose, 0.05, 1000.0);
end




T.FileIO.save("depth.png", T.GL.view_depth_image(d));