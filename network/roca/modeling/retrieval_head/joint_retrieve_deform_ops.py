import torch
from torch import nn
import numpy as np
import h5py

def compute_mahalanobis(query_vecs, mus, sigmas, activation_fn=None, clip_vec=False):
    if not activation_fn == None:
        sigmas = activation_fn(sigmas) + 1.0e-6

    if clip_vec:
        query_vecs = query_vecs.clamp(-100.0, 100.0)

    queries_normalized = torch.square(torch.mul((query_vecs - mus), sigmas))
    distances = torch.sum(queries_normalized, dim= -1)

    return distances

def get_source_latent_codes_fixed(source_labels, latent_codes, device):
    if type(source_labels) is np.ndarray:
        source_labels = torch.from_numpy(source_labels)
    source_labels = source_labels.to(device)
    src_latent_codes = torch.gather(latent_codes, 0, source_labels.unsqueeze(-1).repeat(1,latent_codes.shape[-1]))

    return src_latent_codes

def get_shape(A, param, src_default_param=None, weight=1.0, param_init=None, connectivity_mat=None):
    #batch matrix multiplication
    batch_size = param.shape[0]
    param_dim = param.shape[1]
    param = param.view(batch_size, param_dim, 1)
    src_default_param = src_default_param.view(batch_size, param_dim, 1)

    if (param_init==None):
        param = weight * param
    else:
        param_init = param_init.repeat(batch_size, 1)
        param_init = param_init.view(batch_size, param_dim, 1)
        param = weight * (param - param_init)

    if src_default_param == None:
        if connectivity_mat is None:
            shape = torch.bmm(A, param)
        else:
            param = torch.bmm(connectivity_mat, param)
            shape = torch.bmm(A, param)

    else:
        if connectivity_mat is None:
            shape = torch.bmm(A, param+src_default_param)
        else:
            # print("Using connectivity constraint.")
            param = torch.bmm(connectivity_mat, param+src_default_param)

            shape = torch.bmm(A, param)

    shape = shape.view(batch_size, -1, 3)

    return shape

def regression_loss(embedding_distance, actual_distance, obj_sigmas):
    obj_sigmas = torch.sigmoid(obj_sigmas)
    # obj_sigmas = 1.0
    
    embedding_distance = embedding_distance/100.0
    qij = nn.functional.softmax(-embedding_distance, dim= -1)

    #tranform to reasonable ranges
    actual_distance = actual_distance*100.0

    pij = torch.div(actual_distance, obj_sigmas)
    pij = nn.functional.softmax(-actual_distance, dim= -1)

    # loss = torch.sum(torch.square(pij-qij), dim=-1)
    loss = torch.sum(torch.abs(pij-qij), dim= -1)

    return loss

def get_symmetric(pc):

    reflected_pc = torch.cat([-pc[:,:,0].unsqueeze(-1), pc[:,:,1].unsqueeze(-1), pc[:,:,2].unsqueeze(-1)], axis=2)

    return reflected_pc

def margin_selection(fitting_sorted_idx, embedding_distance, K, num_negs=5):
    #select the positive to be the closest by fitting loss
    positive_idx = fitting_sorted_idx[0,:]

    #random selection of negatives that are "far away"
    perm = torch.randperm(fitting_sorted_idx.size(0)-1) + 1

    negative_idx = fitting_sorted_idx[perm[:num_negs], :]

    #gather corresponding distances
    positive_distances = torch.gather(embedding_distance, 0, positive_idx.unsqueeze(0))
    positive_distances = positive_distances.unsqueeze(0).repeat(num_negs,1,1)
    negative_distances = torch.gather(embedding_distance, 0, negative_idx)

    return positive_distances, negative_distances

def margin_loss(positive_distances, negative_distances, margin, device):
    l = positive_distances - negative_distances + margin
    l = torch.max(l, torch.zeros(l.shape, device=device))
    return l

def get_model(h5_file, semantic=False, mesh=False, constraint=False, pred=False):
    with h5py.File(h5_file, 'r') as f:
        # print(f.keys())
        if pred:
            default_param = f["default_param"][:]
            vertices_mat = f["vertices_mat"][:]
            faces = f["faces"][:]
            constraint_proj_mat = f["constraint_proj_mat"][:]

            return default_param, vertices_mat, faces, constraint_proj_mat


        box_params = f["box_params"][:]
        orig_ids = f["orig_ids"][:]
        default_param = f["default_param"][:]

        ##Point cloud
        points = f["points"][:]
        point_labels = f["point_labels"][:]
        points_mat = f["points_mat"][:]

        if (semantic):
            point_semantic = f["point_semantic"][:]

        if (mesh) :
            vertices = f["vertices"][:]
            vertices_mat = f["vertices_mat"][:]
            faces = f["faces"][:]
            face_labels = f["face_labels"][:]

        if (constraint) :
            constraint_mat = f["constraint_mat"][:]
            constraint_proj_mat = f["constraint_proj_mat"][:]

    if constraint and semantic:
        return box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic, constraint_mat, constraint_proj_mat

    if constraint and mesh:
        return box_params, orig_ids, default_param, points, point_labels, points_mat, vertices, vertices_mat, faces, face_labels, constraint_mat, constraint_proj_mat
    
    if constraint:
        return default_param, points, point_labels, points_mat, constraint_mat, constraint_proj_mat

    if (semantic):
        return box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic

    if (mesh):
        return box_params, orig_ids, default_param, points, point_labels, points_mat, vertices, vertices_mat, faces, face_labels

    else:
        return box_params, orig_ids, default_param, points, point_labels, points_mat

def get_source_info(source_labels, source_model_info, max_num_params, use_connectivity=False):
	'''
	source_labels: contain the labels on which sources are assigned to each target
	source_model_info: dictionary containing the info of the source models such as matrix A and default params
	'''
	source_mats = []
	source_default_params = []
	source_connectivity_mat = []

	for source_label in source_labels:
		points_mat = source_model_info[source_label]["points_mat"]
		padded_mat = np.zeros((points_mat.shape[0], max_num_params))
		padded_mat[0:points_mat.shape[0], 0:points_mat.shape[1]] = points_mat
		# padded_mat = np.expand_dims(padded_mat, axis=0)

		default_param = source_model_info[source_label]["default_param"]
		padded_default_param = np.zeros(max_num_params)
		padded_default_param[:default_param.shape[0]] = default_param
		padded_default_param = np.expand_dims(padded_default_param, axis=0)

		padded_mat = torch.from_numpy(padded_mat).float()
		padded_default_param = torch.from_numpy(padded_default_param).float()

		source_mats.append(padded_mat)
		source_default_params.append(padded_default_param)
		# For connectivity constraint
		if (use_connectivity):
			constraint_proj_mat = source_model_info[source_label]["constraint_proj_mat"]
			constraint_padded_mat = np.zeros((max_num_params, max_num_params))
			constraint_padded_mat[0:constraint_proj_mat.shape[0], 0:constraint_proj_mat.shape[1]] = constraint_proj_mat	

			constraint_padded_mat = torch.from_numpy(constraint_padded_mat).float()
			source_connectivity_mat.append(constraint_padded_mat)

	if (use_connectivity):
		return source_mats, source_default_params, source_connectivity_mat
	else:
		return source_mats, source_default_params, None