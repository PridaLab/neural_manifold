# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:23:28 2022

@author: Usuario
"""

##TO CLEAN!!!

import matplotlib.pyplot as plt
import numpy as np
import os
import pyaldata as pyd

# %%
###########################################################################
#                           PLOT FUNCTIONS
###########################################################################
def plot_2D_embedding_DS(dict_df, embedding_field, gradient_field= "posx", mouse = None, save = False,save_dir = None):
    fnames = list(dict_df.keys())
    emb_s1 = np.concatenate(dict_df[fnames[0]][embedding_field], axis=0)
    if 'pos' in gradient_field:
        pos_s1 = np.concatenate(dict_df[fnames[0]]["pos"], axis=0)
    else:
        pos_s1 = np.concatenate(dict_df[fnames[0]][gradient_field], axis=0)
    if pos_s1.shape[1]>0:
        if 'posx' in gradient_field:
            pos_s1 = pos_s1[:,0].reshape(-1,1)
        elif 'posy' in gradient_field:
            pos_s1 = pos_s1[:,1].reshape(-1,1)
            
    emb_s2 = np.concatenate(dict_df[fnames[1]][embedding_field], axis=0)
    if 'pos' in gradient_field:
        pos_s2 = np.concatenate(dict_df[fnames[1]]["pos"], axis=0)
    else:
        pos_s2 = np.concatenate(dict_df[fnames[1]][gradient_field], axis=0)
    if pos_s2.shape[1]>0:
        if 'posx' in gradient_field:
            pos_s2 = pos_s2[:,0].reshape(-1,1)
        elif 'posy' in gradient_field:
            pos_s2 = pos_s2[:,1].reshape(-1,1)
           
    n_dims = emb_s1.shape[1]
    fig, ax = plt.subplots(n_dims,n_dims,figsize=(n_dims*2,n_dims*2))

    minVal_pos_s1 = np.percentile(pos_s1[:,0], 5)
    maxVal_pos_s1 = np.percentile(pos_s1[:,0], 95)

    minVal_pos_s2 = np.percentile(pos_s2[:,0], 5)
    maxVal_pos_s2 = np.percentile(pos_s2[:,0], 95)
    for ii in range(n_dims):
    	for jj in range(ii+1, n_dims):
            
    		ax[jj,ii].scatter(emb_s1[ :,ii], emb_s1[ :,jj], 
                        c=pos_s1[:,0], vmin=minVal_pos_s1, vmax=maxVal_pos_s1, cmap=plt.cm.magma, s=5, alpha=0.8)
            

    		ax[ii,jj].scatter(emb_s2[ :,ii], emb_s2[ :,jj], 
                        c=pos_s2[:,0], vmin=minVal_pos_s2, vmax=maxVal_pos_s2, cmap=plt.cm.magma, s=5, alpha=0.8)
        
    for ii in range(n_dims):
        ax[ii,ii].scatter(emb_s1[ :,ii], 2+ 0*emb_s1[ :,ii], 
                    c=pos_s1[:,0], vmin=minVal_pos_s1, vmax=maxVal_pos_s1, cmap=plt.cm.magma, s=5, alpha=0.8)
        ax[ii,ii].scatter(emb_s2[ :,ii], 1+ 0*emb_s2[ :,ii], 
                    c=pos_s2[:,0], vmin=minVal_pos_s2, vmax=maxVal_pos_s2, cmap=plt.cm.magma, s=5, alpha=0.8)
        
    for ii in range(n_dims):
       for jj in range(n_dims):     
               ax[ii,jj].get_xaxis().set_ticks([])
               ax[ii,jj].get_yaxis().set_ticks([])
                   
    fig.suptitle(fnames[0][:5]+ '_' +gradient_field)
    fig.tight_layout()
    plt.ion()
    plt.show()   
    if save:
        plt.savefig(os.path.join(save_dir, mouse+'_'+  embedding_field+ '_2D_' + gradient_field))
        
def plot_2D_embedding_LT(dict_df, embedding_field, gradient_field= "posx", mouse = None, save = False,save_dir = None):
    count = 0
    for fname, pd_struct in dict_df.items():
        count +=1
        n_dims = pd_struct[embedding_field][0].shape[1]
        fig, ax = plt.subplots(n_dims,n_dims,figsize=(n_dims*2,n_dims*2))
        struct_left = pyd.select_trials(dict_df[fname], "dir == 'L'")
        emb_left = pyd.concat_trials(struct_left, embedding_field)
        struct_right = pyd.select_trials(dict_df[fname], "dir == 'R'")
        emb_right = pyd.concat_trials(struct_right, embedding_field)
        
        if 'pos' in gradient_field:
            pos_all = np.concatenate(pd_struct["pos"].values, axis=0)
        elif 'index_mat' in gradient_field:
            if 'index_mat' not in pd_struct.columns:
                pd_struct["index_mat"] = [np.zeros((pd_struct["pos"][idx].shape[0],1)).astype(int)+pd_struct["trial_id"][idx] 
                                                 for idx in range(pd_struct.shape[0])]
            pos_all = np.concatenate(pd_struct[gradient_field].values, axis=0)
        else:
            pos_all = np.concatenate(pd_struct[gradient_field].values, axis=0)
        if pos_all.shape[1]>0:
            if 'posx' in gradient_field:
                pos_all = pos_all[:,0].reshape(-1,1)
            elif 'posy' in gradient_field:
                pos_all = pos_all[:,1].reshape(-1,1)
        emb_all = np.concatenate(pd_struct[embedding_field].values, axis=0)
        
       
        minVal_pos = np.percentile(pos_all[:,0], 5)
        maxVal_pos = np.percentile(pos_all[:,0], 95)
        
        for ii in range(n_dims):
        	for jj in range(ii+1, n_dims):
                
        		ax[ii,jj].scatter(emb_left[:,ii], emb_left[ :,jj], 
                            c='C9', s=5, alpha=0.8)
                
        		ax[ii,jj].scatter(emb_right[:,ii], emb_right[ :,jj], 
                            c='C1', s=5, alpha=0.8)
    
        		ax[jj,ii].scatter(emb_all[ :,ii], emb_all[ :,jj], 
                            c=pos_all[:,0], vmin=minVal_pos, vmax=maxVal_pos, cmap=plt.cm.magma, s=5, alpha=0.8)
        for ii in range(n_dims):
            ax[ii,ii].scatter(emb_all[ :,ii], 1+ 0*emb_all[ :,jj], 
                        c=pos_all[:,0], vmin=minVal_pos, vmax=maxVal_pos, cmap=plt.cm.magma, s=5, alpha=0.8)
            
            ax[ii,ii].scatter(emb_left[:,ii], 2 + 0*emb_left[ :,jj], c='C9', s=5, alpha=0.8)
            ax[ii, ii].scatter(emb_right[:,ii], 2+0*emb_right[ :,jj], c='C1', s=5, alpha=0.8)
            
            
        for ii in range(n_dims):
           for jj in range(n_dims):     
                   ax[ii,jj].get_xaxis().set_ticks([])
                   ax[ii,jj].get_yaxis().set_ticks([])
                   
        fig.suptitle(fname)
        fig.tight_layout()
        plt.ion()
        plt.show() 
        if save:
            plt.savefig(os.path.join(save_dir, mouse+'_s'+  str(count) + '_' + embedding_field+ '_2D_' + gradient_field))

#plot 3D points
def plot_3D_embedding_LT(dict_df, embedding_field, gtitle, gradient_field= "pos", dim_to_plot = np.array([0,1,2]).T.reshape(1,-1)):
    n_cols = len(dict_df)    
    fnames = list(dict_df.keys())
    if 'pca' in embedding_field:
        axis_label = 'PC'
    else:
        axis_label = 'Dim'
    n_rows = 2
    if dim_to_plot.shape[0]<n_cols:
        dim_to_plot = np.repeat(dim_to_plot,n_cols, axis=0)
        
    fig = plt.figure(figsize = (16,int(np.ceil(16*n_rows/n_cols))))
    for col, fname in enumerate(fnames):
        struct_left = pyd.select_trials(dict_df[fname], "dir == 'L'")
        concat_emb_left = pyd.concat_trials(struct_left, embedding_field)
        if "time" in gradient_field: 
            struct_left["index_mat"] = [np.zeros((struct_left["pos"][idx].shape[0],1))+struct_left["trial_id"][idx] 
                                                                                for idx in range(struct_left.shape[0])]
            concat_pos_left = pyd.concat_trials(struct_left, "index_mat")
        else:
            concat_pos_left = pyd.concat_trials(struct_left, gradient_field)
        
        struct_right = pyd.select_trials(dict_df[fname], "dir == 'R'")
        concat_emb_right = pyd.concat_trials(struct_right, embedding_field)
        if "time" in gradient_field:
            struct_right["index_mat"] = [np.zeros((struct_right["pos"][idx].shape[0],1))+struct_right["trial_id"][idx] 
                                                                                 for idx in range(struct_right.shape[0])]
            concat_pos_right = pyd.concat_trials(struct_right, "index_mat") 
        else:
            concat_pos_right = pyd.concat_trials(struct_right, gradient_field) 
        
        if "pos" in gradient_field:
            cbar_label = "Position (cm)"
        else:
            cbar_label = gradient_field
            
        ax = plt.subplot(n_rows,n_cols,col+1, projection='3d')
        ax.scatter(*concat_emb_left[:,dim_to_plot[col,:]].T, c= 'C9',edgecolors='face', alpha=0.25, s=40,linewidths=0, label = 'Left')
        ax.scatter(*concat_emb_right[:,dim_to_plot[col,:]].T, c= 'C1',edgecolors='face', alpha=0.25, s=40,linewidths=0, label = 'Right')
        ax.set_xlabel(axis_label + ' ' + str(dim_to_plot[col,0]), size = 16);
        ax.set_ylabel(axis_label + ' ' + str(dim_to_plot[col,1]), size = 16);
        ax.set_zlabel(axis_label + ' ' + str(dim_to_plot[col,2]), size = 16);
        ax.set_title(fname[:21])
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.zaxis.set_tick_params(labelsize=14)
        for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
            axis.line.set_linewidth(2)
            ax.get_xaxis().labelpad = 10
            ax.get_yaxis().labelpad = 10
            ax.get_zaxis().labelpad = 10
            ax.legend()
        
        ax = plt.subplot(n_rows,n_cols,col+n_cols+1, projection='3d')
        p = ax.scatter(*concat_emb_left[:,dim_to_plot[col,:]].T, c= concat_pos_left[:,0],edgecolors='face', 
                                               alpha=0.25, s=40,linewidths=0,cmap=plt.cm.magma)
        ax.scatter(*concat_emb_right[:,dim_to_plot[col,:]].T, c= concat_pos_right[:,0], edgecolors='face', 
                                               alpha=0.25, s=40,linewidths=0,cmap=plt.cm.magma)
        ax.set_xlabel(axis_label + ' ' + str(dim_to_plot[col,0]), size = 16);
        ax.set_ylabel(axis_label + ' ' + str(dim_to_plot[col,1]), size = 16);
        ax.set_zlabel(axis_label + ' ' + str(dim_to_plot[col,2]), size = 16);
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.zaxis.set_tick_params(labelsize=14)
        for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
            axis.line.set_linewidth(2)
            ax.get_xaxis().labelpad = 10
            ax.get_yaxis().labelpad = 10
            ax.get_zaxis().labelpad = 10
            
        # Create an axes for colorbar. The position of the axes is calculated based on the position of ax.
        # You can change 0.01 to adjust the distance between the main image and the colorbar.
        # You can change 0.02 to adjust the width of the colorbar.
        # This practice is universal for both subplots and GeoAxes.
        cbar = fig.colorbar(p, ax=ax,fraction=0.046, pad=0.04)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(cbar_label, rotation=270)
    plt.suptitle(gtitle,fontsize=20)
    fig.tight_layout()
    plt.ion()   
   
#plot 3D points
def plot_3D_embedding_LT_v2(dict_df, embedding_field, gtitle, gradient_field= "pos"):
    n_cols = len(dict_df)    
    fnames = list(dict_df.keys())
    if 'pca' in embedding_field:
        axis_label = 'PC'
    else:
        axis_label = 'Dim'
    n_rows = 2
    fig = plt.figure(figsize = (16,int(np.ceil(16*n_rows/n_cols))))
    for col, fname in enumerate(fnames):
        struct_left = pyd.select_trials(dict_df[fname], "dir == 'L'")
        concat_emb_left = pyd.concat_trials(struct_left, embedding_field)
        struct_left["index_mat"] = [np.zeros((struct_left["pos"][idx].shape[0],1))+struct_left["trial_id"][idx] 
                                                                             for idx in range(struct_left.shape[0])]
        concat_time_left = pyd.concat_trials(struct_left, "index_mat")
        concat_pos_left = pyd.concat_trials(struct_left, gradient_field)
        
        struct_right = pyd.select_trials(dict_df[fname], "dir == 'R'")
        concat_emb_right = pyd.concat_trials(struct_right, embedding_field)
        
        struct_right["index_mat"] = [np.zeros((struct_right["pos"][idx].shape[0],1))+struct_right["trial_id"][idx] 
                                                                                 for idx in range(struct_right.shape[0])]
        concat_time_right = pyd.concat_trials(struct_right, "index_mat") 
        concat_pos_right = pyd.concat_trials(struct_right, gradient_field) 
                    
        ax = plt.subplot(n_rows,n_cols,col+1, projection='3d')
        p = ax.scatter(*concat_emb_left[:,:3].T, c= concat_pos_left[:,0],edgecolors='face', 
                                               alpha=0.5, s=40,linewidths=0,cmap=plt.cm.magma)
        ax.scatter(*concat_emb_right[:,:3].T, c= concat_pos_right[:,0], edgecolors='face', 
                                               alpha=0.5, s=40,linewidths=0,cmap=plt.cm.magma)
        ax.set_xlabel(axis_label + ' 1', size = 16);
        ax.set_ylabel(axis_label + ' 2', size = 16);
        ax.set_zlabel(axis_label + ' 3', size = 16);
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.zaxis.set_tick_params(labelsize=14)
        for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
            axis.line.set_linewidth(2)
            ax.get_xaxis().labelpad = 10
            ax.get_yaxis().labelpad = 10
            ax.get_zaxis().labelpad = 10
        cbar = fig.colorbar(p, ax=ax,fraction=0.046, pad=0.04)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("position (cm)", rotation=270)
        
        
        ax = plt.subplot(n_rows,n_cols,col+n_cols+1, projection='3d')
        p = ax.scatter(*concat_emb_left[:,:3].T, c= concat_time_left[:,0],edgecolors='face', 
                                               alpha=0.5, s=40,linewidths=0,cmap=plt.cm.magma)
        ax.scatter(*concat_emb_right[:,:3].T, c= concat_time_right[:,0], edgecolors='face', 
                                               alpha=0.5, s=40,linewidths=0,cmap=plt.cm.magma)
        ax.set_xlabel(axis_label + ' 1', size = 16);
        ax.set_ylabel(axis_label + ' 2', size = 16);
        ax.set_zlabel(axis_label + ' 3', size = 16);
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.zaxis.set_tick_params(labelsize=14)
        for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
            axis.line.set_linewidth(2)
            ax.get_xaxis().labelpad = 10
            ax.get_yaxis().labelpad = 10
            ax.get_zaxis().labelpad = 10
        cbar = fig.colorbar(p, ax=ax,fraction=0.046, pad=0.04)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel("Time (trial index)", rotation=270)
    plt.suptitle(gtitle,fontsize=20)
    fig.tight_layout()
    plt.ion()
    plt.show()   
    
def plot_trajectory_embedding_LT(dict_df, embedding_field, gtitle):
    n_cols = len(dict_df)    
    fnames = list(dict_df.keys())
    if 'pca' in embedding_field:
        axis_label = 'PC'
    else:
        axis_label = 'Dim'
    fig = plt.figure(figsize = (16,int(np.ceil(16/n_cols))))
    for col, fname in enumerate(fnames):
        ax = plt.subplot(1,n_cols,col+1)
        concat_emb =  pyd.concat_trials(dict_df[fname], embedding_field)
        concat_pos =  pyd.concat_trials(dict_df[fname], "pos")
        av_emb_per_dir = pyd.trial_average(dict_df[fname], "dir")
        ax.scatter(*concat_emb[:,:2].T, c = concat_pos[:,0],edgecolors='face', alpha=0.25, s=40,linewidths=0)
        for idx, dir_emb in enumerate(av_emb_per_dir[embedding_field]):
            if 'L' in av_emb_per_dir["dir"][idx]:
                ax.scatter(dir_emb[:,0], dir_emb[:,1], c='C9', label = 'Left')
                plt.scatter(dir_emb[-1,0], dir_emb[-1,1], color='C0', label='End left')
            elif 'R' in av_emb_per_dir["dir"][idx]:
                ax.scatter(dir_emb[:,0], dir_emb[:,1], c='C1', label = 'Right')
                plt.scatter(dir_emb[-1,0], dir_emb[-1,1], color='C3', label= 'End right')
        ax.set_xlabel(axis_label + ' 1', size = 16);
        ax.set_ylabel(axis_label + ' 2', size = 16);
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        for axis in ['left', 'bottom']:
            ax.spines[axis].set_linewidth(2)
        for axis in ['top', 'right']:
            ax.spines[axis].set_visible(False)
        ax.get_xaxis().labelpad = 10
        ax.get_yaxis().labelpad = 10
        ax.legend()
    plt.suptitle(gtitle,fontsize=20)
    fig.tight_layout()
    plt.ion()
    plt.show()        

def plot_PCA_variance(models_pca,max_pc, max_cumu_pc, pca_dims, gtitle):
    n_cols = len(models_pca)    
    fnames = list(models_pca.keys())
    fig, ax = plt.subplots(figsize = (16, 4),ncols = n_cols, nrows = 1)
    x_space = np.linspace(1, pca_dims, pca_dims)
    for col, fname in enumerate(fnames):
        ax[col].bar(x_space, 100*models_pca[fname].explained_variance_ratio_[:pca_dims])
        ax[col].set_xlabel('PC #', size=16)
        ax[col].set_ylabel('Explained Variance (%)', color = 'C2',size=16)
        ax[col].set_xlim([0.2,pca_dims+0.8])
        ax[col].set_ylim([0, max_pc])
        ax[col].xaxis.set_tick_params(labelsize=16)
        ax[col].yaxis.set_tick_params(labelsize=16)
        ax[col].set_title(fname[:21])
        ax[col].spines['right'].set_visible(False)
        ax[col].spines['top'].set_visible(False)
        ax2 = ax[col].twinx()
        ax2.plot(x_space, 100*np.cumsum(models_pca[fname].explained_variance_ratio_[:pca_dims]), color = 'k')
        ax2.set_ylabel('Cumulative Variance (%)', size=16)
        ax2.xaxis.set_tick_params(labelsize=16)
        ax2.yaxis.set_tick_params(labelsize=16)
        ax2.set_ylim([0, max_cumu_pc])
    plt.suptitle(gtitle, fontsize=20)
    fig.tight_layout()
    plt.ion()
    plt.show()
    
def plot_decoder_LT(dict_df, gtitle, x_dims, color_code = ['C7', 'C4', 'C3', 'C0'], y_dim = 0, sem=True):
    lines_names = list(dict_df.keys())

    files_names = list(dict_df[lines_names[0]].keys())
    n_files = len(files_names)
    
    decoders_names = list(dict_df[lines_names[0]][files_names[0]].keys())
    n_decoders = len(decoders_names)
    
    fig, ax = plt.subplots(figsize = (16,12), ncols = n_files, nrows = n_decoders)
    for row, decoder in enumerate(decoders_names):
        for col, file in enumerate(files_names): 
            for nline, line_name in enumerate(lines_names):
                temp = np.copy(dict_df[line_name][file][decoder][y_dim, :,:])
                temp[temp<-1e-7] = np.nan
                m = np.nanmean(temp, axis=1)
                if sem:
                    sd = np.nanstd(temp, axis=1)/np.sqrt(temp.shape[1] - np.count_nonzero(np.isnan(temp),axis=1))
                else:
                    sd = np.nanstd(temp, axis=1)
                if m.shape[0]==1:
                    x_dims = 10
                    ax[row,col].plot([1, x_dims], [m,m], color = color_code[nline], label = line_name, linestyle = '--')
                    ax[row,col].fill_between([1, x_dims], m-sd, m+sd, color = color_code[nline], alpha=0.25)
                else:
                    x_dims = np.where(m>0)[0][-1]+1
                    x_space = np.linspace(1, x_dims, x_dims)

                    ax[row,col].plot(x_space, m[:x_dims], color = color_code[nline], label = line_name)
                    ax[row,col].fill_between(x_space, m[:x_dims]-sd[:x_dims], m[:x_dims]+sd[:x_dims], 
                                             color = color_code[nline], alpha=0.25)
            ax[row,col].set_ylim([0,1])
            ax[row,col].set_xlim([x_space[0]-0.5, x_space[-1]+0.5])
            ax[row,col].set_title(decoder + '-' + file[:21])
            plt.setp(ax[row,col].spines.values(), linewidth=2)
            ax[row,col].spines['right'].set_visible(False)
            ax[row,col].spines['top'].set_visible(False)        

    ax[row,col].legend(fontsize=16)
    plt.suptitle(gtitle, fontsize=20)
    plt.tight_layout()
    plt.ion()
    plt.show()

def plot_internal_dim(internal_dim_dict,fnames,gtitle, m = None):
    minx = np.Inf
    miny = np.Inf
    maxy = np.NINF
    maxx = np.NINF
    if m == None:
        m = []
        for file in fnames:
            m.append(internal_dim_dict[file + "_internal_dim"])
        m = np.nanmean(m)
        if isinstance(m,np.ndarray):
            m = m[0][0]
    plt.figure()
    for file in fnames:
        plt.plot(internal_dim_dict[file + "_radii_vs_nn"][:,0],  internal_dim_dict[file + "_radii_vs_nn"][:,1], label = file)           
        minx = np.min([minx, internal_dim_dict[file + "_radii_vs_nn"][0,0]])
        miny = np.min([miny, np.min(internal_dim_dict[file + "_radii_vs_nn"][:,1])])
        maxy = np.max([maxy, np.max(internal_dim_dict[file + "_radii_vs_nn"][:,1])])
        maxx = np.max([maxx, internal_dim_dict[file + "_radii_vs_nn"][-1,0]])

    ns = np.linspace(miny-np.floor(maxx)*m, maxy, 8)
    for n in ns:
        x = np.linspace(np.max([((miny-n)/m), minx]), np.min([((maxy-n)/m),maxx]), 20).reshape(-1,1)
        y = m*x + n
        if n==ns[0]:
            plt.plot(x,y, color = [.5,.5,.5], linestyle = '--', label = "slope of "+ str(m))
        else:
            plt.plot(x,y, color = [.5,.5,.5], linestyle = '--')
    plt.xlim([np.max([((miny-n)/m), minx]), maxx])
    plt.ylim([miny, maxy])
    plt.legend()
    plt.title(gtitle)
    plt.ion()
    plt.show()
    