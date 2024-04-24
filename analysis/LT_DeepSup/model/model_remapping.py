num_iters = 5
folder_name = f"0{int(100*field_type_probs['same_prob'])}same_" + \
                f"0{int(100*field_type_probs['xmirror_prob'])}xmirror_" + \
                f"0{int(100*field_type_probs['remap_prob'])}remap_" + \
                f"0{int(100*field_type_probs['remap_no_field_prob'])}remapnofield_" + \
                f"0{int(100*field_type_probs['no_field_prob'])}nofield_"  + \
                f"0{int(100*noise)}noise"

save_dir = os.path.join(base_save_dir, folder_name)
if not os.path.exists(save_dir):
        os.mkdir(save_dir)


allo_prob = 0

print(f"{allo_prob:.2f} ALLOCENTRIC")
iteration_name = f"0{int(100*allo_prob)}allocentric" + \
                f"_0{int(100*noise)}noise"
save_fig_dir = os.path.join(save_dir, iteration_name)
if not os.path.exists(save_fig_dir):
        os.mkdir(save_fig_dir)

try:
    results = load_pickle(save_dir, 'statistic_model_results.pkl')
    print("loading old results")
except:
    print("no old results")
    results = {}


remaining_prob = 1 - allo_prob
local_prob_list = np.arange(0.1, remaining_prob+0.05, 0.05)[::-1]

rot_angles = np.zeros((len(local_prob_list),num_iters))
rot_angles2 = np.zeros((len(local_prob_list),num_iters))
rot_distances = np.zeros((len(local_prob_list),num_iters))
rot_distances2 = np.zeros((len(local_prob_list),num_iters))
for idx, local_prob in enumerate(local_prob_list):
    cell_type_probs = {
        'local_anchored_prob': local_prob,
        'allo_prob': allo_prob,
    }
    remap_prob = 1 - np.sum([x for n,x in cell_type_probs.items()])
    cell_type_probs['remap_prob'] = remap_prob*0.1
    cell_type_probs['remap_no_field_prob'] = remap_prob*0.9

    for it in range(num_iters):
        model_test = StatisticModel(pos, direction_mat, rot_pos,rot_direction_mat, num_cells = num_cells)
        model_test.compute_fields(**field_type_probs)
        model_test.compute_cell_types(**cell_type_probs)
        model_test.compute_rotation_fields()

        model_test.compute_traces(noise_sigma = noise)
        model_test.compute_rotation_traces(rot_pos,rot_direction_mat)
        model_test.clean_traces()

        model_test.compute_umap_both()
        model_test.clean_umap()


        model_test.compute_rotation()
        model_test.compute_distance()

        a = model_test.find_rotation(model_test.cent_pre[::2], model_test.aligned_cent_rot[::2],model_test.norm_vector_pre)
        new_angle = np.abs(model_test.angles[np.argmin(a)])*180/np.pi
        b = model_test.find_rotation(model_test.cent_pre[1::2], model_test.aligned_cent_rot[1::2],model_test.norm_vector_pre)
        new_angle2 = np.abs(model_test.angles[np.argmin(b)])*180/np.pi
        while np.abs(model_test.rot_angle - np.mean([new_angle, new_angle2]))>50:
            print(f"\t\terror in iteration {it}/{num_iters}: {model_test.rot_angle:.2f}º - {model_test.remap_dist:.2f} dist")
            model_test = StatisticModel(pos, direction_mat, rot_pos,rot_direction_mat, num_cells = num_cells)
            model_test.compute_fields(**field_type_probs)
            model_test.compute_cell_types(**cell_type_probs)
            model_test.compute_rotation_fields()
            model_test.compute_traces(noise_sigma = noise)
            model_test.compute_rotation_traces(rot_pos,rot_direction_mat)
            model_test.clean_traces()
            model_test.compute_umap_both()
            model_test.clean_umap()
            model_test.compute_rotation()
            a = model_test.find_rotation(model_test.cent_pre[::2], model_test.aligned_cent_rot[::2],model_test.norm_vector_pre)
            new_angle = np.abs(model_test.angles[np.argmin(a)])*180/np.pi
            b = model_test.find_rotation(model_test.cent_pre[1::2], model_test.aligned_cent_rot[1::2],model_test.norm_vector_pre)
            new_angle2 = np.abs(model_test.angles[np.argmin(b)])*180/np.pi
            model_test.compute_distance()

        fig = model_test.plot_rotation()
        plt.savefig(os.path.join(save_fig_dir,f'model_{int(100*remap_prob)}remap_{it}iter_rotation.png'), dpi = 400,bbox_inches="tight")
        plt.close(fig)

        fig = model_test.plot_distance()
        plt.savefig(os.path.join(save_fig_dir,f'model_{int(100*remap_prob)}remap_{it}iter_distance.png'), dpi = 400,bbox_inches="tight")
        plt.close(fig)

        rot_angles[idx, it] = model_test.rot_angle
        rot_distances[idx, it] = model_test.plane_remap_dist
        rot_distances2[idx, it] = model_test.remap_dist
        a = model_test.find_rotation(model_test.cent_pre, model_test.cent_rot,model_test.norm_vector_pre)
        rot_angles2[idx, it] = np.abs(model_test.angles[np.argmin(a)])*180/np.pi

        print(f"remap_prob: {remap_prob:.2f} {it+1}/{num_iters}: {rot_angles[idx,it]:.2f}º ({rot_angles2[idx,it]:.2f}) - {rot_distances[idx,it]:.2f} ({rot_distances2[idx,it]:.2f}) dist")


m = np.nanmean(rot_distances, axis=1)
sd = np.nanstd(rot_distances, axis=1)

x_space = local_prob_list[::-1]
fig = plt.figure()
plt.plot(x_space, m)
plt.fill_between(x_space, m-sd, m+sd, alpha = 0.3)
plt.ylabel('Remap Dist Planes')
plt.xlabel('Proportion of remap cells')
plt.savefig(os.path.join(save_fig_dir,'rotation_plot_remap_plane.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_fig_dir,'rotation_plot_remap_plane.png'), dpi = 400,bbox_inches="tight")
plt.close(fig)


m = np.nanmean(rot_distances2, axis=1)
sd = np.nanstd(rot_distances2, axis=1)

x_space = local_prob_list[::-1]
fig = plt.figure()
plt.plot(x_space, m)
plt.fill_between(x_space, m-sd, m+sd, alpha = 0.3)
plt.ylabel('Remap Dist')
plt.xlabel('Proportion of remap cells')
plt.savefig(os.path.join(save_fig_dir,'rotation_plot_remap.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_fig_dir,'rotation_plot_remap.png'), dpi = 400,bbox_inches="tight")
plt.close(fig)


results['allo'][(allo_prob, noise)] = {'rot_angles': rot_angles, 'rot_angles_no_alignment': rot_angles2,
                        'rot_distances': rot_distances, 'rot_distances_no_ellipse': rot_distances2, 
                        'local_prob_list': local_prob_list, 'num_cells': num_cells, 
                        'field_type_probs': field_type_probs, 'cell_type_probs': cell_type_probs}

with open(os.path.join(save_dir,'statistic_model_results.pkl'), 'wb') as f:
    pickle.dump(results, f)

##########################################################################################################################

num_iters = 10
folder_name = f"0{int(100*field_type_probs['same_prob'])}same_" + \
                f"0{int(100*field_type_probs['xmirror_prob'])}xmirror_" + \
                f"0{int(100*field_type_probs['remap_prob'])}remap_" + \
                f"0{int(100*field_type_probs['remap_no_field_prob'])}remapnofield_" + \
                f"0{int(100*field_type_probs['no_field_prob'])}nofield_"  + \
                f"0{int(100*noise)}noise"

save_dir = os.path.join(base_save_dir, folder_name)
if not os.path.exists(save_dir):
        os.mkdir(save_dir)




local_prob = 0.1
print(f"{local_prob:.2f} LOCAL-CUE")
iteration_name = f"0{int(100*allo_prob)}localcue" + \
                f"_0{int(100*noise)}noise"
save_fig_dir = os.path.join(save_dir, iteration_name)
if not os.path.exists(save_fig_dir):
        os.mkdir(save_fig_dir)

try:
    results2 = load_pickle(save_dir, 'statistic_model_results.pkl')
    print("loading old results")
except:
    print("no old results")
    results = {}


remaining_prob = 1 - local_prob
allo_prob_list = np.arange(0.2, remaining_prob+0.05, 0.05)[::-1]

rot_angles = np.zeros((len(allo_prob_list),num_iters))
rot_angles2 = np.zeros((len(allo_prob_list),num_iters))
rot_distances = np.zeros((len(allo_prob_list),num_iters))
rot_distances2 = np.zeros((len(allo_prob_list),num_iters))
for idx, allo_prob in enumerate(allo_prob_list):
    cell_type_probs = {
        'local_anchored_prob': local_prob,
        'allo_prob': allo_prob,
    }
    remap_prob = 1 - np.sum([x for n,x in cell_type_probs.items()])
    cell_type_probs['remap_prob'] = remap_prob*0.1
    cell_type_probs['remap_no_field_prob'] = remap_prob*0.9

    for it in range(num_iters):
        model_test = StatisticModel(pos, direction_mat, rot_pos,rot_direction_mat, num_cells = num_cells)
        model_test.compute_fields(**field_type_probs)
        model_test.compute_cell_types(**cell_type_probs)
        model_test.compute_rotation_fields()

        model_test.compute_traces(noise_sigma = noise)
        model_test.compute_rotation_traces(rot_pos,rot_direction_mat)
        model_test.clean_traces()

        model_test.compute_umap_both()
        model_test.clean_umap()


        model_test.compute_rotation()
        model_test.compute_distance()

        a = model_test.find_rotation(model_test.cent_pre[::2], model_test.aligned_cent_rot[::2],model_test.norm_vector_pre)
        new_angle = np.abs(model_test.angles[np.argmin(a)])*180/np.pi
        b = model_test.find_rotation(model_test.cent_pre[1::2], model_test.aligned_cent_rot[1::2],model_test.norm_vector_pre)
        new_angle2 = np.abs(model_test.angles[np.argmin(b)])*180/np.pi
        while np.abs(model_test.rot_angle - np.mean([new_angle, new_angle2]))>50:
            print(f"\t\terror in iteration {it}/{num_iters}: {model_test.rot_angle:.2f}º - {model_test.remap_dist:.2f} dist")
            model_test = StatisticModel(pos, direction_mat, rot_pos,rot_direction_mat, num_cells = num_cells)
            model_test.compute_fields(**field_type_probs)
            model_test.compute_cell_types(**cell_type_probs)
            model_test.compute_rotation_fields()
            model_test.compute_traces(noise_sigma = noise)
            model_test.compute_rotation_traces(rot_pos,rot_direction_mat)
            model_test.clean_traces()
            model_test.compute_umap_both()
            model_test.clean_umap()
            model_test.compute_rotation()
            a = model_test.find_rotation(model_test.cent_pre[::2], model_test.aligned_cent_rot[::2],model_test.norm_vector_pre)
            new_angle = np.abs(model_test.angles[np.argmin(a)])*180/np.pi
            b = model_test.find_rotation(model_test.cent_pre[1::2], model_test.aligned_cent_rot[1::2],model_test.norm_vector_pre)
            new_angle2 = np.abs(model_test.angles[np.argmin(b)])*180/np.pi
            model_test.compute_distance()

        fig = model_test.plot_rotation()
        plt.savefig(os.path.join(save_fig_dir,f'model_{int(100*remap_prob)}remap_{it}iter_rotation.png'), dpi = 400,bbox_inches="tight")
        plt.close(fig)

        fig = model_test.plot_distance()
        plt.savefig(os.path.join(save_fig_dir,f'model_{int(100*remap_prob)}remap_{it}iter_distance.png'), dpi = 400,bbox_inches="tight")
        plt.close(fig)

        rot_angles[idx, it] = model_test.rot_angle
        rot_distances[idx, it] = model_test.plane_remap_dist
        rot_distances2[idx, it] = model_test.remap_dist
        a = model_test.find_rotation(model_test.cent_pre, model_test.cent_rot,model_test.norm_vector_pre)
        rot_angles2[idx, it] = np.abs(model_test.angles[np.argmin(a)])*180/np.pi

        print(f"remap_prob: {remap_prob:.2f} {it+1}/{num_iters}: {rot_angles[idx,it]:.2f}º ({rot_angles2[idx,it]:.2f}) - {rot_distances[idx,it]:.2f} ({rot_distances2[idx,it]:.2f}) dist")


m = np.nanmean(rot_distances, axis=1)
sd = np.nanstd(rot_distances, axis=1)

x_space = allo_prob_list[::-1]
fig = plt.figure()
plt.plot(x_space, m)
plt.fill_between(x_space, m-sd, m+sd, alpha = 0.3)
plt.ylabel('Remap Dist Planes')
plt.xlabel('Proportion of remap cells')
plt.savefig(os.path.join(save_fig_dir,'rotation_plot_remap_plane.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_fig_dir,'rotation_plot_remap_plane.png'), dpi = 400,bbox_inches="tight")
plt.close(fig)


m = np.nanmean(rot_distances2, axis=1)
sd = np.nanstd(rot_distances2, axis=1)

x_space = allo_prob_list[::-1]
fig = plt.figure()
plt.plot(x_space, m)
plt.fill_between(x_space, m-sd, m+sd, alpha = 0.3)
plt.ylabel('Remap Dist')
plt.xlabel('Proportion of remap cells')
plt.savefig(os.path.join(save_fig_dir,'rotation_plot_remap.svg'), dpi = 400,bbox_inches="tight")
plt.savefig(os.path.join(save_fig_dir,'rotation_plot_remap.png'), dpi = 400,bbox_inches="tight")
plt.close(fig)




results['local'][(allo_prob, noise)] = {'rot_angles': rot_angles, 'rot_angles_no_alignment': rot_angles2,
                        'rot_distances': rot_distances, 'rot_distances_no_ellipse': rot_distances2, 
                        'allo_prob_list': allo_prob_list, 'num_cells': num_cells, 
                        'field_type_probs': field_type_probs, 'cell_type_probs': cell_type_probs}

with open(os.path.join(save_dir,'statistic_model_results.pkl'), 'wb') as f:
    pickle.dump(results, f)
