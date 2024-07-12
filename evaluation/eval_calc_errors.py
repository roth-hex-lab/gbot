# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Calculates error of 6D object pose estimates."""

import os
import time
import argparse
import copy
import numpy as np
import math
from bop_toolkit.bop_toolkit_lib import config
from bop_toolkit.bop_toolkit_lib import dataset_params
from bop_toolkit.bop_toolkit_lib import inout
from bop_toolkit.bop_toolkit_lib import misc
from bop_toolkit.bop_toolkit_lib import pose_error
from bop_toolkit.bop_toolkit_lib import renderer


def VOCap(rec, prec):
  idx = np.where(rec != np.inf)
  if len(idx[0]) == 0:
    return 0
  rec = rec[idx]
  prec = prec[idx]
  mrec = np.array([0.0] + list(rec) + [0.1])
  mpre = np.array([0.0] + list(prec) + [prec[-1]])
  for i in range(1, prec.shape[0]):
    mpre[i] = max(mpre[i], mpre[i - 1])
  i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
  ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) * 10
  return ap


def cal_auc(add_dis, threshold=0.1):
  D = np.array(add_dis)/1000
  count = (D < threshold).sum()
  aps = count/len(add_dis)
  return aps * 100

# PARAMETERS (can be overwritten by the command line arguments below).
################################################################################
p = {
  # Top N pose estimates (with the highest score) to be evaluated for each
  # object class in each image.
  'assembly_ids': ['Hand_ScrewClamp'],
  'track_modes':['GBOT'],
  'scene_ids':['normal', 'dynamic', 'hand', 'blur'],
  # Options: 0 = all, -1 = given by the number of GT poses.
  'n_top': 1,

  # Pose error function.
  # Options: 'vsd', 'mssd', 'mspd', 'ad', 'adi', 'add', 'cus', 're', 'te, etc.
  'error_types': ['ad', 'te', 're'],

  # VSD parameters.
  'vsd_deltas': {
    'hb': 15,
    'icbin': 15,
    'icmi': 15,
    'itodd': 5,
    'lm': 15,
    'lmo': 15,
    'ruapc': 15,
    'tless': 15,
    'tudl': 15,
    'tyol': 15,
    'ycbv': 15,
    'hope': 15,
    '3Dprint': 15,
  },
  'vsd_taus': list(np.arange(0.05, 0.51, 0.05)),
  'vsd_normalized_by_diameter': True,

  # MSSD/MSPD parameters (see misc.get_symmetry_transformations).
  'max_sym_disc_step': 0.01,

  # Whether to ignore/break if some errors are missing.
  'skip_missing': True,

  # Type of the renderer (used for the VSD pose error function).
  'renderer_type': 'python',  # Options: 'vispy', 'cpp', 'python'.

  # Names of files with results for which to calculate the errors (assumed to be
  # stored in folder p['results_path']). See docs/bop_challenge_2019.md for a
  # description of the format. Example results can be found at:
  # https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019/


  # Folder containing the BOP datasets.
  #'datasets_path': config.datasets_path,
  'datasets_path': 'path2your/GBOT_dataset',

  # File with a list of estimation targets to consider. The file is assumed to
  # be stored in the dataset folder.

  # Template of path to the output file with calculated errors.
  'out_errors_path': os.path.join(
    '{eval_path}', '{result_name}', '{error_sign}',
    'errors_{scene_id:s}.json')
}
################################################################################


# Command line arguments.
# ------------------------------------------------------------------------------
vsd_deltas_str =\
  ','.join(['{}:{}'.format(k, v) for k, v in p['vsd_deltas'].items()])

parser = argparse.ArgumentParser()
parser.add_argument('--n_top', default=p['n_top'])
parser.add_argument('--vsd_deltas', default=vsd_deltas_str)
parser.add_argument('--vsd_taus', default=','.join(map(str, p['vsd_taus'])))
parser.add_argument('--vsd_normalized_by_diameter',
                    default=p['vsd_normalized_by_diameter'])
parser.add_argument('--max_sym_disc_step', default=p['max_sym_disc_step'])
parser.add_argument('--skip_missing', default=p['skip_missing'])
parser.add_argument('--renderer_type', default=p['renderer_type'])
parser.add_argument('--datasets_path', default=p['datasets_path'])
parser.add_argument('--out_errors_path', default=p['out_errors_path'])
args = parser.parse_args()

p['n_top'] = int(args.n_top)
p['vsd_deltas'] = {str(e.split(':')[0]): float(e.split(':')[1])
                   for e in args.vsd_deltas.split(',')}
p['vsd_taus'] = list(map(float, args.vsd_taus.split(',')))
p['vsd_normalized_by_diameter'] = bool(args.vsd_normalized_by_diameter)
p['max_sym_disc_step'] = float(args.max_sym_disc_step)
p['skip_missing'] = bool(args.skip_missing)
p['renderer_type'] = str(args.renderer_type)
p['datasets_path'] = str(args.datasets_path)
p['out_errors_path'] = str(args.out_errors_path)

misc.log('-----------')
misc.log('Parameters:')
for k, v in p.items():
  misc.log('- {}: {}'.format(k, v))
misc.log('-----------')
eval_dict = {'GearedCaliper': ['Caliper_Fix', 'Caliper_Move_Bottom', 'Caliper_Move_Top_Vernier'],
             'HobbyCornerClamp': ['Corner_Clamp_Base', 'Corner_Clamp_Bolt', 'Corner_Clamp_Jaw'],
             'NanoViseV2': ['NanoViseV2_CLAMP_BALLJOINT', 'NanoViseV2_CLAMP_BASE', 'NanoViseV2_CLAMP_SCREW',
                            'NanoViseV2_VISE_BASE', 'NanoViseV2_VISE_SCREW', 'NanoViseV2_VISE_SLIDER', 'NanoViseV2_CLAMP_HEADPLATE', 'NanoViseV2_CLAMP_NUT'],
             'Hand_ScrewClamp': ['HS_Jaw1', 'HS_Jaw2', 'HS_Knob1', 'HS_Knob2', 'HS_Pad', 'HS_Thread1', 'HS_Thread2'],
             'Liftpod': ['Liftpod_armfirst', 'Liftpod_armlast', 'Liftpod_bar', 'Liftpod_baseplate',
                         'Liftpod_clampframe', 'Liftpod_clampslider', 'Liftpod_sleeve']}
# Error calculation.
# ------------------------------------------------------------------------------
for assembly_id in p['assembly_ids']:
  for track_mode in p['track_modes']:
    for scene_id in p['scene_ids']:
      for error_type in p['error_types']:
        result_filename = os.path.join(p['datasets_path'], assembly_id, 'test', track_mode + scene_id + '_assembly-test.csv')
        eval_path = os.path.join(p['datasets_path'], assembly_id, 'eval_' + track_mode)
        misc.log('Processing: {}'.format(result_filename))
        targets_filename ='eval_'+ scene_id + '.json'
        ests_counter = 0
        time_start = time.time()

        # Parse info about the method and the dataset from the filename.
        result_name = os.path.splitext(os.path.basename(result_filename))[0]
        result_info = result_name.split('_')
        method = str(result_info[0])
        dataset_info = result_info[1].split('-')
        dataset = str(dataset_info[0])
        split = str(dataset_info[1])
        split_type = str(dataset_info[2]) if len(dataset_info) > 2 else None
        split_type_str = ' - ' + split_type if split_type is not None else ''

        # Load dataset parameters.
        dp_split = dataset_params.get_split_params(
          os.path.join(p['datasets_path'], assembly_id), dataset, split, split_type)

        model_type = 'eval'
        dp_model = dataset_params.get_model_params(
          os.path.join(p['datasets_path'], assembly_id), dataset, model_type)

        # Load object models.
        models = {}
        #if p['error_type'] in ['ad', 'add', 'adi', 'mssd', 'mspd', 'proj']:
        misc.log('Loading object models...')
        for obj_id in dp_model['obj_ids']:
            models[obj_id] = inout.load_ply(dp_model['model_dir'].format(obj_id=obj_id))

        # Load models info.
        models_info = None
        #if p['error_type'] in ['ad', 'add', 'adi', 'vsd', 'mssd', 'mspd', 'cus', 're', 'te']:
        models_info = inout.load_json(dp_model['models_info_dir'], keys_to_int=True)

        # Get sets of symmetry transformations for the object models.
        models_sym = None
        if error_type in ['mssd', 'mspd']:
          models_sym = {}
          for obj_id in dp_model['obj_ids']:
            models_sym[obj_id] = misc.get_symmetry_transformations(
              models_info[obj_id], p['max_sym_disc_step'])

        # Initialize a renderer.
        ren = None
        if error_type in ['vsd', 'cus']:
          misc.log('Initializing renderer...')
          width, height = dp_split['im_size']
          ren = renderer.create_renderer(
            width, height, p['renderer_type'], mode='depth')
          for obj_id in dp_model['obj_ids']:
            ren.add_object(obj_id, dp_model['model_dir'].format(obj_id=obj_id))

        # Load the estimation targets.
        targets = inout.load_json(
          os.path.join(p['datasets_path'], assembly_id, targets_filename))

        # Organize the targets by scene, image and object.
        misc.log('Organizing estimation targets...')
        targets_org = {}
        for target in targets:
          targets_org.setdefault(target['scene_id'], {}).setdefault(
            target['im_id'], {})[target['obj_id']] = target

        # Load pose estimates.
        misc.log('Loading pose estimates...')
        ests = inout.load_bop_results(result_filename)

        # Organize the pose estimates by scene, image and object.
        misc.log('Organizing pose estimates...')
        ests_org = {}
        for est in ests:
          ests_org.setdefault(est['scene_id'], {}).setdefault(
            est['im_id'], {}).setdefault(est['obj_id'], []).append(est)

        for scene_id, scene_targets in targets_org.items():

          # Load camera and GT poses for the current scene.
          scene_camera = inout.load_scene_camera(
            dp_split['scene_camera_dir'].format(scene_id=scene_id))
          scene_gt = inout.load_scene_gt(dp_split['scene_gt_dir'].format(
            scene_id=scene_id))

          scene_errs = []

          for im_ind, (im_id, im_targets) in enumerate(scene_targets.items()):

            if im_ind % 10 == 0:
              misc.log(
                'Calculating error {} - method: {}, dataset: {}{}, scene: {}, '
                'im: {}'.format(
                  error_type, method, dataset, split_type_str, scene_id, im_ind))

            # Intrinsic camera matrix.
            K = scene_camera[0]['cam_K']

            # Load the depth image if VSD is selected as the pose error function.
            depth_im = None
            if error_type == 'vsd':
              depth_path = dp_split['depth_path'].format(
                scene_id=scene_id, im_id=im_id)
              depth_im = inout.load_depth(depth_path)
              depth_im *= scene_camera[im_id]['depth_scale']  # Convert to [mm].

            for obj_id, target in im_targets.items():

              # The required number of top estimated poses.
              if p['n_top'] == 0:  # All estimates are considered.
                n_top_curr = None
              elif p['n_top'] == -1:  # Given by the number of GT poses.
                # n_top_curr = sum([gt['obj_id'] == obj_id for gt in scene_gt[im_id]])
                n_top_curr = target['inst_count']
              else:
                n_top_curr = p['n_top']

              # Get the estimates.
              try:
                obj_test = ests_org[scene_id][im_id]
                obj_ests = ests_org[scene_id][im_id][obj_id]
                obj_count = len(obj_ests)
              except KeyError:
                obj_ests = []
                obj_count = 0

              # Check the number of estimates.
              if not p['skip_missing'] and obj_count < n_top_curr:
                raise ValueError(
                  'Not enough estimates for scene: {}, im: {}, obj: {} '
                  '(provided: {}, expected: {})'.format(
                    scene_id, im_id, obj_id, obj_count, n_top_curr))

              # Sort the estimates by score (in descending order).
              obj_ests_sorted = sorted(
                enumerate(obj_ests), key=lambda x: x[1]['score'], reverse=True)

              # Select the required number of top estimated poses.
              obj_ests_sorted = obj_ests_sorted[slice(0, n_top_curr)]
              ests_counter += len(obj_ests_sorted)

              # Calculate error of each pose estimate w.r.t. all GT poses of the same
              # object class.
              for est_id, est in obj_ests_sorted:

                # Estimated pose.
                R_e = est['R']
                t_e = est['t']

                errs = {}  # Errors w.r.t. GT poses of the same object class.
                for gt_id, gt in enumerate(scene_gt[im_id]):
                  if gt['obj_id'] != obj_id:
                    continue

                  # Ground-truth pose.
                  R_g = gt['cam_R_m2c']
                  t_g = gt['cam_t_m2c']

                  # Check if the projections of the bounding spheres of the object in
                  # the two poses overlap (to speed up calculation of some errors).
                  sphere_projections_overlap = None
                  if error_type in ['vsd', 'cus']:
                    radius = 0.5 * models_info[obj_id]['diameter']
                    sphere_projections_overlap = misc.overlapping_sphere_projections(
                      radius, t_e.squeeze(), t_g.squeeze())

                  # Check if the bounding spheres of the object in the two poses
                  # overlap (to speed up calculation of some errors).
                  spheres_overlap = None
                  if error_type in ['ad', 'add', 'adi', 'mssd']:
                    center_dist = np.linalg.norm(t_e - t_g)
                    spheres_overlap = center_dist < models_info[obj_id]['diameter']

                  if error_type == 'vsd':
                    if not sphere_projections_overlap:
                      e = [1.0] * len(p['vsd_taus'])
                    else:
                      e = pose_error.vsd(
                        R_e, t_e, R_g, t_g, depth_im, K, p['vsd_deltas'][dataset],
                        p['vsd_taus'], p['vsd_normalized_by_diameter'],
                        models_info[obj_id]['diameter'], ren, obj_id, 'step')

                  elif error_type == 'mssd':
                    if not spheres_overlap:
                      e = [float('inf')]
                    else:
                      e = [pose_error.mssd(
                        R_e, t_e, R_g, t_g, models[obj_id]['pts'],
                        models_sym[obj_id])]

                  elif error_type == 'mspd':
                    e = [pose_error.mspd(
                      R_e, t_e, R_g, t_g, K, models[obj_id]['pts'],
                      models_sym[obj_id])]

                  elif error_type in ['ad', 'add', 'adi']:
                    if not spheres_overlap:
                      # Infinite error if the bounding spheres do not overlap. With
                      # typically used values of the correctness threshold for the AD
                      # error (e.g. k*diameter, where k = 0.1), such pose estimates
                      # would be considered incorrect anyway.
                      e = [float('inf')]
                    else:
                      if error_type == 'ad':
                        if obj_id in dp_model['symmetric_obj_ids']:
                          e = [pose_error.adi(
                            R_e, t_e, R_g, t_g, models[obj_id]['pts'])]
                        else:
                          e = [pose_error.add(
                            R_e, t_e, R_g, t_g, models[obj_id]['pts'])]

                      elif error_type == 'add':
                        e = [pose_error.add(
                          R_e, t_e, R_g, t_g, models[obj_id]['pts'])]

                      else:  # 'adi'
                        e = [pose_error.adi(
                          R_e, t_e, R_g, t_g, models[obj_id]['pts'])]

                  elif error_type == 'cus':
                    if sphere_projections_overlap:
                      e = [pose_error.cus(
                        R_e, t_e, R_g, t_g, K, ren, obj_id)]
                    else:
                      e = [1.0]

                  elif error_type == 'proj':
                    e = [pose_error.proj(
                      R_e, t_e, R_g, t_g, K, models[obj_id]['pts'])]

                  elif error_type == 'rete':
                    e = [pose_error.re(R_e, R_g), pose_error.te(t_e, t_g)]

                  elif error_type == 're':
                    e = [pose_error.re(R_e, R_g)]

                  elif error_type == 'te':
                    e = [pose_error.te(t_e, t_g)]

                  else:
                    raise ValueError('Unknown pose error function.')

                  errs[gt_id] = e

                # Save the calculated errors.
                scene_errs.append({
                  'im_id': im_id,
                  'obj_id': obj_id,
                  'est_id': est_id,
                  'score': est['score'],
                  'errors': errs
                })

          def save_errors(_error_sign, _scene_errs):
            # Save the calculated errors to a JSON file.
            errors_path = p['out_errors_path'].format(
              eval_path=eval_path, result_name=result_name,
              error_sign=_error_sign, scene_id=scene_id)
            misc.ensure_dir(os.path.dirname(errors_path))
            misc.log('Saving errors to: {}'.format(errors_path))
            inout.save_json(errors_path, _scene_errs)

          # Save the calculated errors.
          if error_type == 'vsd':

            # For VSD, save errors for each tau value to a different file.
            for vsd_tau_id, vsd_tau in enumerate(p['vsd_taus']):
              error_sign = misc.get_error_signature(
                error_type, p['n_top'], vsd_delta=p['vsd_deltas'][dataset],
                vsd_tau=vsd_tau)

              # Keep only errors for the current tau.
              scene_errs_curr = copy.deepcopy(scene_errs)
              for err in scene_errs_curr:
                for gt_id in err['errors'].keys():
                  err['errors'][gt_id] = [err['errors'][gt_id][vsd_tau_id]]

              save_errors(error_sign, scene_errs_curr)
          else:
            error_sign = misc.get_error_signature(error_type, p['n_top'])
            save_errors(error_sign, scene_errs)

          # Calculate the performance scores.

          obj4eval = eval_dict[assembly_id]

          message_sum = '_____________\n Assembly: %s Track mode: %s Scene: %s\n' % (assembly_id, method, scene_id)
          if error_type == 'ad':
            error_dict = {}
            for obj_idm in models_info.keys():
              error_dict[obj_idm] = []
            for err in scene_errs:
              for gt_id in err['errors'].keys():
                error_dict[err['obj_id']] += err['errors'][gt_id]

            ADD_sum = 0
            for obj_idm in models_info.keys():
              ADD = cal_auc(error_dict[obj_idm], threshold = 0.1)
              message = 'ADD-s of %s: %f \n' % (obj_idm, ADD)
              message_sum += message
              print(message)
              if obj_idm in obj4eval:
                if math.isfinite(ADD):
                  ADD_sum += ADD
            ADD_ave = ADD_sum/ len(obj4eval)
            message = 'Average ADD-S of the selected objects is %f \n' % ADD_ave
            message_sum += message
            print(message)
          if error_type == 're':
            unsymmetric_obj = []
            for obj_eval in obj4eval:
              if obj_eval not in dp_model['symmetric_obj_ids']:
                unsymmetric_obj.append(obj_eval)
            error_dict = {}
            for obj_idm in models_info.keys():
              error_dict[obj_idm] = []
            re_list = []
            re_num = 0.0
            for err in scene_errs:
              for gt_id in err['errors'].keys():
                error_dict[err['obj_id']] += err['errors'][gt_id]
            re_sum = 0
            for obj_idm in models_info.keys():
              try:
                re = sum(error_dict[obj_idm])/len(error_dict[obj_idm])
              except:
                pass
              message = 'rotation error of %s in degree: %f \n' % (obj_idm, re)
              message_sum += message
              print(message)
              if obj_idm in unsymmetric_obj:
                re_sum += re
            re_ave = re_sum / len(unsymmetric_obj)
            message = 'Average rotation error  of the selected objects in degree is %f \n' % re_ave
            message_sum += message
            print(message)


          if error_type == 'te':
            error_dict = {}
            for obj_idm in models_info.keys():
              error_dict[obj_idm] = []
            te_list = []
            te_num = 0.0
            for err in scene_errs:
              for gt_id in err['errors'].keys():
                if math.isfinite(sum(err['errors'][gt_id])) and sum(err['errors'][gt_id]) < 1000:
                  error_dict[err['obj_id']] += err['errors'][gt_id]
            te_sum = 0
            for obj_idm in models_info.keys():
              try:
                te = sum(error_dict[obj_idm]) / len(error_dict[obj_idm])
              except:
                pass
              message = 'Translation error of %s in mm: %f \n' % (obj_idm, te)
              message_sum += message
              print(message)
              if obj_idm in obj4eval:
                te_sum += te
            te_ave = te_sum / len(obj4eval)
            message = 'Average translation error  of the selected objects in mm is %f \n' % te_ave
            message_sum += message
            print(message)


        time_total = time.time() - time_start
        misc.log('Calculation of errors for {} estimates took {}s.'.format(
          ests_counter, time_total))

        misc.log('Done.')
        result_path = os.path.join(p['datasets_path'], assembly_id, 'test', 'eval_results.txt')
        with open(result_path, "a+") as text_file:
          text_file.write(message_sum)
        misc.log('Performance results saved to '+ result_path)
