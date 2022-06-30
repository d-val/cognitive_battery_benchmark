
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

MV.setup_visualizer()

constraints = Gen.choicemap()
constraints[:N] = length(shape_models)
for i=1:length(shape_models)
    constraints[i => :shape] = i
    constraints[1 => i => :pose] = initial_poses[i]
end
constraints[1 => :obs] = T.voxelize(hcat(entities_over_time[t]...), resolution)

@time trace, _ = Gen.generate(multiobject_tracking_init_state, (1, renderer, length(shape_models), resolution), constraints);
MV.reset_visualizer()
MV.viz(Gen.get_retval(trace).voxelized_point_clouds[1] ./10.0; color=T.I.colorant"black", channel_name=:gen_cloud)
MV.viz(Gen.get_retval(trace).observations[1] ./10.0; color=T.I.colorant"red", channel_name=:obs_cloud)


new_args = (2, renderer, length(shape_models), resolution)
new_arg_diff = Tuple([Gen.UnknownChange() for _=1:length(new_args)]);

constraints[2 => :obs] = T.voxelize(hcat(entities_over_time[t+1]...), resolution)
@time trace, _ = Gen.update(trace, new_args, new_arg_diff, constraints);


MV.reset_visualizer()
MV.viz(Gen.get_retval(trace).voxelized_point_clouds[2] ./10.0; color=T.I.colorant"black", channel_name=:gen_cloud)
MV.viz(Gen.get_retval(trace).observations[2] ./10.0; color=T.I.colorant"red", channel_name=:obs_cloud)

MV.reset_visualizer()
MV.viz(shape_models[3] ./10.0; color=T.I.colorant"black", channel_name=:gen_cloud)

for object_id in 1:5
    for _ in 1:100
        trace, acc = T.drift_move(trace, 2 => object_id => :pose, 0.05, 1000.0);
    end
end


MV.reset_visualizer()
MV.viz(Gen.get_retval(trace).voxelized_point_clouds[2] ./10.0; color=T.I.colorant"black", channel_name=:gen_cloud)
MV.viz(Gen.get_retval(trace).observations[2] ./10.0; color=T.I.colorant"red", channel_name=:obs_cloud)