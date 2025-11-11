# https://stackoverflow.com/questions/62806175/xarray-combine-by-coords-return-the-monotonic-global-index-error
# https://github.com/pydata/xarray/issues/8828

from contextlib import nullcontext
import dask
from dask.diagnostics import ProgressBar
import logging
import multiview_stitcher
from multiview_stitcher import registration, vis_utils
from multiview_stitcher.mv_graph import NotEnoughOverlapError
from multiview_stitcher.registration import get_overlap_bboxes
import os.path
import shutil
import xarray as xr

from src.Timer import Timer
from src.image.Video import Video
from src.image.flatfield import flatfield_correction
from src.image.ome_helper import save_image, exists_output_image
from src.image.ome_tiff_helper import save_tiff
from src.image.source_helper import create_dask_source
from src.image.util import *
from src.metrics import calc_ncc, calc_ssim
from src.util import *


dask.config.set(scheduler='threads')


class MVSRegistration:
    def __init__(self, params_general):
        super().__init__()
        self.params_general = params_general

        params_logging = self.params_general.get('logging', {})
        self.verbose = params_logging.get('verbose', False)
        self.logging_dask = params_logging.get('dask', False)
        self.logging_time = params_logging.get('time', False)
        self.ui = self.params_general.get('ui', '')
        self.mpl_ui = ('mpl' in self.ui or 'plot' in self.ui)
        self.napari_ui = ('napari' in self.ui)
        self.source_transform_key = 'source_metadata'
        self.reg_transform_key = 'registered'
        self.transition_transform_key = 'transition'

        logging.info(f'Multiview-stitcher version: {multiview_stitcher.__version__}')

    def run_operation(self, fileset_label, filenames, params, global_rotation=None, global_center=None):
        self.fileset_label = fileset_label
        self.filenames = filenames
        self.file_labels = get_unique_file_labels(filenames)
        self.params = params
        self.global_rotation = global_rotation
        self.global_center = global_center

        input_dir = os.path.dirname(filenames[0])
        parts = split_numeric_dict(filenames[0])
        output_pattern = params['output'].format_map(parts)
        self.output = os.path.join(input_dir, output_pattern)    # preserve trailing slash: do not use os.path.normpath()

        with ProgressBar(minimum=10, dt=1) if self.logging_dask else nullcontext():
            return self._run_operation()

    def _run_operation(self):
        params = self.params
        filenames = self.filenames
        file_labels = self.file_labels
        output = self.output

        operation = params['operation']
        overlap_threshold = params.get('overlap_threshold', 0.5)
        source_metadata = import_metadata(params.get('source_metadata', {}), input_path=params['input'])
        save_images = params.get('save_images', True)
        target_scale = params.get('scale')
        extra_metadata = import_metadata(params.get('extra_metadata', {}), input_path=params['input'])
        channels = extra_metadata.get('channels', [])
        normalise_orientation = 'norm' in source_metadata

        show_original = self.params_general.get('show_original', False)
        output_params = self.params_general.get('output', {})
        clear = output_params.get('clear', False)
        overwrite = output_params.get('overwrite', True)

        is_stack = ('stack' in operation)
        is_3d = ('3d' in operation)
        is_simple_stack = is_stack and not is_3d
        is_transition = ('transition' in operation)
        is_channel_overlay = (len(channels) > 1)

        mappings_header = ['id','x_pixels', 'y_pixels', 'z_pixels', 'x', 'y', 'z', 'rotation']

        if len(filenames) == 0:
            logging.warning('Skipping (no images)')
            return False

        registered_fused_filename = output + 'registered'
        mappings_filename = os.path.join(output, params.get('mappings', 'mappings.json'))

        output_dir = os.path.dirname(output)
        if not overwrite and exists_output_image(registered_fused_filename, output_params.get('format')):
            logging.warning(f'Skipping existing output {os.path.normpath(output_dir)}')
            return False
        if clear:
            shutil.rmtree(output_dir, ignore_errors=True)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with Timer('init sims', self.logging_time):
            sims, scales, positions, rotations = self.init_sims(target_scale=target_scale)

        with Timer('pre-process', self.logging_time):
            sims, register_sims, indices = self.preprocess(sims, params)

        data = []
        for label, sim, scale in zip(file_labels, sims, scales):
            position, rotation = get_data_mapping(sim, transform_key=self.source_transform_key)
            position_pixels = np.array(position) / scale
            row = [label] + list(position_pixels) + list(position) + [rotation]
            data.append(row)
        export_csv(output + 'prereg_mappings.csv', data, header=mappings_header)

        if show_original:
            # before registration:
            logging.info('Exporting original...')
            original_positions_filename = output + 'positions_original.pdf'

            with Timer('plot positions', self.logging_time):
                vis_utils.plot_positions(sims, transform_key=self.source_transform_key,
                                         use_positional_colors=False, view_labels=file_labels, view_labels_size=3,
                                         show_plot=self.mpl_ui, output_filename=original_positions_filename)

            if self.napari_ui:
                shapes = [get_sim_shape_2d(sim, transform_key=self.source_transform_key) for sim in sims]
                self.update_napari_signal.emit(f'{self.fileset_label} original', shapes, file_labels)

            if save_images:
                if output_params.get('thumbnail'):
                    with Timer('create thumbnail', self.logging_time):
                        self.save_thumbnail(output + 'thumb_original',
                                            nom_sims=sims,
                                            transform_key=self.source_transform_key)

                original_fused = self.fuse(sims, transform_key=self.source_transform_key)

                original_fused_filename = output + 'original'
                save_image(original_fused_filename, output_params.get('format'), original_fused, channels=channels,
                           transform_key=self.source_transform_key, params=output_params)

        if len(filenames) == 1 and save_images:
            logging.warning('Skipping registration (single image)')
            save_image(registered_fused_filename, output_params.get('format'), sims[0], channels=channels,
                       translation0=positions[0], params=output_params)
            return False

        is_simple_stack = is_stack and not is_3d
        _, has_overlaps = self.validate_overlap(sims, file_labels, is_simple_stack, is_simple_stack or is_channel_overlay)
        overall_overlap = np.mean(has_overlaps)
        if overall_overlap < overlap_threshold:
            raise ValueError(f'Not enough overlap: {overall_overlap * 100:.1f}%')

        if not overwrite and os.path.exists(mappings_filename):
            logging.info('Loading registration mappings...')
            # load registration mappings
            mappings = import_json(mappings_filename)
            # copy transforms to sims
            for sim, label in zip(sims, file_labels):
                mapping = param_utils.affine_to_xaffine(np.array(mappings[label]))
                if is_stack:
                    transform = param_utils.identity_transform(ndim=3)
                    transform.loc[{dim: mapping.coords[dim] for dim in mapping.dims}] = mapping
                else:
                    transform = mapping
                si_utils.set_sim_affine(sim, transform, transform_key=self.reg_transform_key)
        else:
            with Timer('register', self.logging_time):
                results = self.register(sims, register_sims, indices, params)

            reg_result = results['reg_result']
            sims = results['sims']

            logging.info('Exporting registered...')
            metrics = self.calc_metrics(results, file_labels)
            mappings = metrics['mappings']
            logging.info(metrics['summary'])
            export_json(mappings_filename, mappings)
            export_json(output + 'metrics.json', metrics)
            data = []
            for sim, (label, mapping), scale, position, rotation in zip(sims, mappings.items(), scales, positions, rotations):
                if not normalise_orientation:
                    # rotation already in msim affine transform
                    rotation = None
                position, rotation = get_data_mapping(sim, transform_key=self.reg_transform_key,
                                                      transform=np.array(mapping),
                                                      translation0=position,
                                                      rotation=rotation)
                position_pixels = np.array(position) / scale
                row = [label] + list(position_pixels) + list(position) + [rotation]
                data.append(row)
            export_csv(output + 'mappings.csv', data, header=mappings_header)

            for reg_label, reg_item in reg_result.items():
                if isinstance(reg_item, dict):
                    summary_plot = reg_item.get('summary_plot')
                    if summary_plot is not None:
                        figure, axes = summary_plot
                        summary_plot_filename = output + f'{reg_label}.pdf'
                        figure.savefig(summary_plot_filename)

        registered_positions_filename = output + 'positions_registered.pdf'
        with Timer('plot positions', self.logging_time):
            vis_utils.plot_positions(sims, transform_key=self.reg_transform_key,
                                     use_positional_colors=False, view_labels=file_labels, view_labels_size=3,
                                     show_plot=self.mpl_ui, output_filename=registered_positions_filename)

        if self.napari_ui:
            shapes = [get_sim_shape_2d(sim, transform_key=self.reg_transform_key) for sim in sims]
            self.update_napari_signal.emit(f'{self.fileset_label} registered', shapes, file_labels)

        if save_images:
            if output_params.get('thumbnail'):
                with Timer('create thumbnail', self.logging_time):
                    self.save_thumbnail(output + 'thumb', nom_sims=sims, transform_key=self.reg_transform_key)

            with Timer('fuse image', self.logging_time):
                fused_image = self.fuse(sims)

            logging.info('Saving fused image...')
            with Timer('save fused image', self.logging_time):
                save_image(registered_fused_filename, output_params.get('format'), fused_image, channels=channels,
                           transform_key=self.reg_transform_key, translation0=positions[0], params=output_params)

        if is_transition:
            self.save_video(output, sims, fused_image)

        return True

    def init_sims(self, target_scale=None):
        operation = self.params['operation']
        source_metadata = import_metadata(self.params.get('source_metadata', 'source'), input_path=self.params['input'])
        chunk_size = self.params_general.get('chunk_size', [1024, 1024])
        extra_metadata = import_metadata(self.params.get('extra_metadata', {}), input_path=self.params['input'])
        z_scale = extra_metadata.get('scale', {}).get('z')

        logging.info('Initialising sims...')
        sources = [create_dask_source(file, source_metadata) for file in self.filenames]
        source0 = sources[0]
        images = []
        sims = []
        scales = []
        translations = []
        rotations = []

        is_stack = ('stack' in operation)
        has_z_size = (source0.get_size().get('z', 0) > 0)
        is_3d = (has_z_size or '3d' in operation)
        pyramid_level = 0

        output_order = 'zyx' if is_stack or is_3d else 'yx'
        ndims = len(output_order)
        if source0.get_nchannels() > 1:
            output_order += 'c'

        last_z_position = None
        different_z_positions = False
        delta_zs = []
        for filename, source in zip(self.filenames, sources):
            scale = source.get_pixel_size()
            translation = source.get_position()
            rotation = source.get_rotation()

            if target_scale:
                pyramid_level = np.argmin(abs(np.array(source.scales) - target_scale))
                pyramid_scale = source.scales[pyramid_level]
                scale = {dim: size * pyramid_scale if dim in 'xy' else size for dim, size in scale.items()}
            if 'invert' in source_metadata:
                translation[0] = -translation[0]
                translation[1] = -translation[1]
            if len(translation) >= 3:
                z_position = translation['z']
            else:
                z_position = 0
            if last_z_position is not None and z_position != last_z_position:
                different_z_positions = True
                delta_zs.append(z_position - last_z_position)
            if self.global_rotation is not None:
                rotation = self.global_rotation

            dask_data = source.get_data(level=pyramid_level)
            image = redimension_data(dask_data, source.dimension_order, output_order)

            scales.append(scale)
            translations.append(translation)
            rotations.append(rotation)
            images.append(image)
            last_z_position = z_position

        if z_scale is None:
            if len(delta_zs) > 0:
                z_scale = np.min(delta_zs)
            else:
                z_scale = 1

        if 'norm' in source_metadata:
            size = np.array(source0.get_size()) * source0.get_pixel_size_micrometer()
            center = None
            if 'center' in source_metadata:
                if 'global' in source_metadata:
                    center = self.global_center
                else:
                    center = np.mean(translations, 0)
            elif 'origin' in source_metadata:
                center = np.zeros(ndims)
            translations, rotations = normalise_rotated_positions(translations, rotations, size, center)

        #translations = [np.array(translation) * 1.25 for translation in translations]

        increase_z_positions = is_stack and not different_z_positions
        z_position = 0
        scales2 = []
        translations2 = []
        for source, image, scale, translation, rotation, file_label in zip(sources, images, scales, translations, rotations, self.file_labels):
            # transform #dimensions need to match
            if len(scale) > 0 and 'z' not in scale:
                scale['z'] = abs(z_scale)
            if (len(translation) > 0 and 'z' not in translation) or increase_z_positions:
                translation['z'] = z_position
            if increase_z_positions:
                z_position += z_scale
            channel_labels = [channel.get('label', '') for channel in source.get_channels()]
            if rotation is None or 'norm' in source_metadata:
                # if positions are normalised, don't use rotation
                transform = None
            else:
                transform = param_utils.invert_coordinate_order(
                    create_transform(translation, rotation, matrix_size=ndims + 1)
                )
            if file_label in extra_metadata:
                transform2 = extra_metadata[file_label]
                if transform is None:
                    transform = np.array(transform2)
                else:
                    transform = np.array(combine_transforms([transform, transform2]))
            sim = si_utils.get_sim_from_array(
                image,
                dims=list(output_order),
                scale=scale,
                translation=translation,
                affine=transform,
                transform_key=self.source_transform_key,
                c_coords=channel_labels
            )
            if len(sim.chunksizes.get('x')) == 1 and len(sim.chunksizes.get('y')) == 1:
                sim = sim.chunk(xyz_to_dict(chunk_size))
            sims.append(sim)
            scales2.append(dict_to_xyz(scale))
            translations2.append(dict_to_xyz(translation))
        return sims, scales2, translations2, rotations

    def validate_overlap(self, sims, labels, is_stack=False, expect_large_overlap=False):
        min_dists = []
        has_overlaps = []
        n = len(sims)
        positions = [get_sim_position_final(sim) for sim in sims]
        sizes = [np.linalg.norm(get_sim_physical_size(sim)) for sim in sims]
        for i in range(n):
            norm_dists = []
            # check if only single z slices
            if is_stack:
                if i + 1 < n:
                    compare_indices = [i + 1]
                else:
                    compare_indices = []
            else:
                compare_indices = range(n)
            for j in compare_indices:
                if not j == i:
                    distance = math.dist(positions[i], positions[j])
                    norm_dist = distance / np.mean([sizes[i], sizes[j]])
                    norm_dists.append(norm_dist)
            if len(norm_dists) > 0:
                norm_dist = min(norm_dists)
                min_dists.append(float(norm_dist))
                if norm_dist >= 1:
                    logging.warning(f'{labels[i]} has no overlap')
                    has_overlaps.append(False)
                elif expect_large_overlap and norm_dist > 0.5:
                    logging.warning(f'{labels[i]} has small overlap')
                    has_overlaps.append(False)
                else:
                    has_overlaps.append(True)
        return min_dists, has_overlaps

    def preprocess(self, sims, params):
        flatfield_quantiles = params.get('flatfield_quantiles')
        normalisation = params.get('normalisation', '')
        filter_foreground = params.get('filter_foreground', False)

        if filter_foreground:
            foreground_map = calc_foreground_map(sims)
        else:
            foreground_map = None
        if flatfield_quantiles is not None:
            logging.info('Flat-field correction...')
            new_sims = [None] * len(sims)
            for sim_indices in group_sims_by_z(sims):
                sims_z_set = [sims[i] for i in sim_indices]
                foreground_map_z_set = [foreground_map[i] for i in sim_indices] if foreground_map is not None else None
                new_sims_z_set = flatfield_correction(sims_z_set, self.source_transform_key, flatfield_quantiles,
                                                      foreground_map=foreground_map_z_set)
                for sim_index, sim in zip(sim_indices, new_sims_z_set):
                    new_sims[sim_index] = sim
            sims = new_sims

        if normalisation:
            use_global = ('global' in normalisation)
            if use_global:
                logging.info('Normalising (global)...')
            else:
                logging.info('Normalising (individual)...')
            new_sims = normalise(sims, self.source_transform_key, use_global=use_global)
        else:
            new_sims = sims

        if filter_foreground:
            logging.info('Filtering foreground images...')
            #tile_vars = np.array([np.asarray(np.std(sim)).item() for sim in sims])
            #threshold1 = np.mean(tile_vars)
            #threshold2 = np.median(tile_vars)
            #threshold3, _ = cv.threshold(np.array(tile_vars).astype(np.uint16), 0, 1, cv.THRESH_OTSU)
            #threshold = min(threshold1, threshold2, threshold3)
            #foregrounds = (tile_vars >= threshold)
            new_sims = [sim for sim, is_foreground in zip(new_sims, foreground_map) if is_foreground]
            logging.info(f'Foreground images: {len(new_sims)} / {len(sims)}')
            indices = np.where(foreground_map)[0]
        else:
            indices = range(len(sims))
        return sims, new_sims, indices

    def register(self, sims, register_sims, indices, params):
        sim0 = sims[0]
        ndims = si_utils.get_ndim_from_sim(sim0)

        operation = params['operation']
        reg_params = params.get('method')
        if isinstance(reg_params, dict):
            reg_method = reg_params.get('name', '').lower()
        else:
            reg_method = reg_params.lower()
        use_orthogonal_pairs = params.get('use_orthogonal_pairs', False)

        is_stack = ('stack' in operation)
        is_3d = ('3d' in operation)
        debug = self.params_general.get('debug', False)

        reg_channel = params.get('channel', 0)
        if isinstance(reg_channel, int):
            reg_channel_index = reg_channel
            reg_channel = None
        else:
            reg_channel_index = None

        groupwise_resolution_kwargs = {
            'transform': params.get('transform_type')  # options include 'translation', 'rigid', 'affine'
        }
        pairwise_reg_func_kwargs = None
        if is_stack and not is_3d:
            # register in 2d; pairwise consecutive views
            register_sims = [si_utils.max_project_sim(sim, dim='z') for sim in register_sims]
            pairs = [(index, index + 1) for index in range(len(register_sims) - 1)]
        elif use_orthogonal_pairs:
            origins = np.array([get_sim_position_final(sim) for sim in register_sims])
            size = get_sim_physical_size(sim0)
            pairs, _ = get_orthogonal_pairs(origins, size)
            logging.info(f'#pairs: {len(pairs)}')
            for pair in pairs:
                print(f'{self.file_labels[pair[0]]} - {self.file_labels[pair[1]]}')
        else:
            pairs = None

        if is_3d:
            overlap_tolerance = {'z': 1}
        else:
            overlap_tolerance = None

        if '3din2d' in reg_method:
            from src.registration_methods.RegistrationMethodANTs3Din2D import RegistrationMethodANTs3Din2D
            registration_method = RegistrationMethodANTs3Din2D(sim0, reg_params, debug)
            pairwise_reg_func = registration_method.registration
        elif 'cpd' in reg_method:
            from src.registration_methods.RegistrationMethodCPD import RegistrationMethodCPD
            registration_method = RegistrationMethodCPD(sim0, reg_params, debug)
            pairwise_reg_func = registration_method.registration
        elif 'feature' in reg_method or 'orb' in reg_method or 'sift' in reg_method:
            if 'cv' in reg_method:
                from src.registration_methods.RegistrationMethodCvFeatures import RegistrationMethodCvFeatures
                registration_method = RegistrationMethodCvFeatures(sim0, reg_params, debug)
            else:
                from src.registration_methods.RegistrationMethodSkFeatures import RegistrationMethodSkFeatures
                registration_method = RegistrationMethodSkFeatures(sim0, reg_params, debug)
            pairwise_reg_func = registration_method.registration
        elif 'ant' in reg_method:
            pairwise_reg_func = registration.registration_ANTsPy
            # args for ANTsPy registration: used internally by ANYsPy algorithm
            pairwise_reg_func_kwargs = {
                'transform_types': ['Rigid'],
                "aff_random_sampling_rate": 0.5,
                "aff_iterations": (2000, 2000, 1000, 1000),
                "aff_smoothing_sigmas": (4, 2, 1, 0),
                "aff_shrink_factors": (16, 8, 2, 1),
            }
        else:
            pairwise_reg_func = registration.phase_correlation_registration

        # Pass registration through metrics method
        #from src.registration_methods.RegistrationMetrics import RegistrationMetrics
        #registration_metrics = RegistrationMetrics(sim0, pairwise_reg_function)
        #pairwise_reg_function = registration_metrics.registration
        # TODO: extract metrics from registration_metrics

        logging.info(f'Registration method: {reg_method}')

        try:
            logging.info('Registering...')
            register_msims = [msi_utils.get_msim_from_sim(sim) for sim in register_sims]
            reg_result = registration.register(
                register_msims,
                reg_channel=reg_channel,
                reg_channel_index=reg_channel_index,
                transform_key=self.source_transform_key,
                new_transform_key=self.reg_transform_key,

                pairs=pairs,
                pre_registration_pruning_method=None,

                pairwise_reg_func=pairwise_reg_func,
                pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,
                groupwise_resolution_kwargs=groupwise_resolution_kwargs,

                post_registration_do_quality_filter=True,
                post_registration_quality_threshold=0.1,

                plot_summary=self.mpl_ui,
                return_dict=True,

                overlap_tolerance=overlap_tolerance,
            )
            # copy transforms from register sims to unmodified sims
            for reg_msim, index in zip(register_msims, indices):
                si_utils.set_sim_affine(
                    sims[index],
                    msi_utils.get_transform_from_msim(reg_msim, transform_key=self.reg_transform_key),
                    transform_key=self.reg_transform_key)

            # set missing transforms
            for sim in sims:
                if self.reg_transform_key not in si_utils.get_tranform_keys_from_sim(sim):
                    si_utils.set_sim_affine(
                        sim,
                        param_utils.identity_transform(ndim=ndims, t_coords=[0]),
                        transform_key=self.reg_transform_key)

            mappings = reg_result['params']
            # re-index from subset of sims
            residual_error_dict = reg_result.get('groupwise_resolution', {}).get('metrics', {}).get('residuals', {})
            residual_error_dict = {(indices[key[0]], indices[key[1]]): value.item()
                                   for key, value in residual_error_dict.items()}
            registration_qualities_dict = reg_result.get('pairwise_registration', {}).get('metrics', {}).get('qualities', {})
            registration_qualities_dict = {(indices[key[0]], indices[key[1]]): value
                                           for key, value in registration_qualities_dict.items()}
        except NotEnoughOverlapError:
            logging.warning('Not enough overlap')
            reg_result = {}
            mappings = [param_utils.identity_transform(ndim=ndims, t_coords=[0])] * len(sims)
            residual_error_dict = {}
            registration_qualities_dict = {}

        # re-index from subset of sims
        mappings_dict = {index: mapping for index, mapping in zip(indices, mappings)}

        if is_stack:
            # set 3D affine transforms from 2D registration params
            for index, sim in enumerate(sims):
                # check if already 3D
                if 4 not in si_utils.get_affine_from_sim(sim, transform_key=self.reg_transform_key).shape:
                    affine_3d = param_utils.identity_transform(ndim=3)
                    affine_3d.loc[{dim: mappings[index].coords[dim] for dim in mappings[index].sel(t=0).dims}] = mappings[index].sel(t=0)
                    si_utils.set_sim_affine(sim, affine_3d, transform_key=self.reg_transform_key)

        return {'reg_result': reg_result,
                'mappings': mappings_dict,
                'residual_errors': residual_error_dict,
                'registration_qualities': registration_qualities_dict,
                'sims': sims,
                'pairs': pairs}

    def fuse(self, sims, transform_key=None):
        if transform_key is None:
            transform_key = self.reg_transform_key
        operation = self.params['operation']
        extra_metadata = import_metadata(self.params.get('extra_metadata', {}), input_path=self.params['input'])
        channels = extra_metadata.get('channels', [])
        z_scale = extra_metadata.get('scale', {}).get('z')
        if z_scale is None:
            if 'z' in sims[0].dims:
                z_scale = np.min(np.diff(sorted(set([si_utils.get_origin_from_sim(sim).get('z', 0) for sim in sims]))))
        if not z_scale:
            z_scale = 1

        is_3d = ('3d' in operation)
        is_channel_overlay = (len(channels) > 1)

        sim0 = sims[0]
        source_type = sim0.dtype

        output_stack_properties = calc_output_properties(sims, transform_key, z_scale=z_scale)
        if is_channel_overlay:
            # convert to multichannel images
            if self.verbose:
                logging.info(f'Output stack: {output_stack_properties}')
            data_size = np.prod(list(output_stack_properties['shape'].values())) * len(sims) * source_type.itemsize
            logging.info(f'Fusing channels {print_hbytes(data_size)}')

            channel_sims = [fusion.fuse(
                [sim],
                transform_key=transform_key,
                output_stack_properties=output_stack_properties
            ) for sim in sims]
            channel_sims = [sim.assign_coords({'c': [channels[simi]['label']]}) for simi, sim in enumerate(channel_sims)]
            fused_image = xr.combine_nested([sim.rename() for sim in channel_sims], concat_dim='c', combine_attrs='override')
        else:
            if is_3d:
                z_positions = sorted(set([si_utils.get_origin_from_sim(sim).get('z', 0) for sim in sims]))
                z_shape = len(z_positions)
                if z_shape <= 1:
                    z_shape = len(sims)
                output_stack_properties['shape']['z'] = z_shape
            if self.verbose:
                logging.info(f'Output stack: {output_stack_properties}')
            data_size = np.prod(list(output_stack_properties['shape'].values())) * source_type.itemsize
            logging.info(f'Fusing {print_hbytes(data_size)}')

            fused_image = fusion.fuse(
                sims,
                transform_key=transform_key,
                output_stack_properties=output_stack_properties,
                fusion_func=fusion.simple_average_fusion,
            )
        return fused_image

    def save_thumbnail(self, output_filename, nom_sims=None, transform_key=None):
        extra_metadata = import_metadata(self.params.get('extra_metadata', {}), input_path=self.params['input'])
        channels = extra_metadata.get('channels', [])
        output_params = self.params_general['output']
        thumbnail_scale = output_params.get('thumbnail_scale', 16)
        sims = self.init_sims(target_scale=thumbnail_scale)[0]

        if nom_sims is not None:
            if sims[0].sizes['x'] >= nom_sims[0].sizes['x']:
                logging.warning('Unable to generate scaled down thumbnail due to lack of source pyramid sizes')
                return

            if transform_key is not None and transform_key != self.source_transform_key:
                for nom_sim, sim in zip(nom_sims, sims):
                    si_utils.set_sim_affine(sim,
                                            si_utils.get_affine_from_sim(nom_sim, transform_key=transform_key),
                                            transform_key=transform_key)
        fused_image = self.fuse(sims, transform_key=transform_key).squeeze()
        save_image(output_filename, output_params.get('thumbnail'), fused_image, channels=channels,
                   transform_key=transform_key, params=output_params)

    def calc_overlap_metrics(self, results):
        nccs = {}
        ssims = {}
        sims = results['sims']
        pairs = results['pairs']
        if pairs is None:
            origins = np.array([get_sim_position_final(sim) for sim in sims])
            size = get_sim_physical_size(sims[0])
            pairs, _ = get_orthogonal_pairs(origins, size)
        for pair in pairs:
            try:
                # experimental; in case fail to extract overlap images
                overlap_sims = self.get_overlap_images((sims[pair[0]], sims[pair[1]]), self.reg_transform_key)
                nccs[pair] = calc_ncc(overlap_sims[0], overlap_sims[1])
                ssims[pair] = calc_ssim(overlap_sims[0], overlap_sims[1])
                #frcs[pair] = calc_frc(overlap_sims[0], overlap_sims[1])
            except Exception as e:
                logging.exception(e)
                #logging.warning(f'Failed to calculate resolution metric')
        return {'ncc': nccs, 'ssim': ssims}

    def get_overlap_images(self, sims, transform_key):
        # functionality copied from registration.register_pair_of_msims()
        spatial_dims = si_utils.get_spatial_dims_from_sim(sims[0])
        overlap_tolerance = {dim: 0.0 for dim in spatial_dims}
        overlap_sims = []
        for sim in sims:
            if 't' in sim.coords.xindexes:
                # work-around for points error in get_overlap_bboxes()
                sim1 = si_utils.sim_sel_coords(sim, {'t': 0})
            else:
                sim1 = sim
            overlap_sims.append(sim1)
        lowers, uppers = get_overlap_bboxes(
            overlap_sims[0],
            overlap_sims[1],
            input_transform_key=transform_key,
            output_transform_key=None,
            overlap_tolerance=overlap_tolerance,
        )

        reg_sims_spacing = [
            si_utils.get_spacing_from_sim(sim) for sim in sims
        ]

        tol = 1e-6
        overlaps_sims = [
            sim.sel(
                {
                    # add spacing to include bounding pixels
                    dim: slice(
                        lowers[isim][idim] - tol - reg_sims_spacing[isim][dim],
                        uppers[isim][idim] + tol + reg_sims_spacing[isim][dim],
                    )
                    for idim, dim in enumerate(spatial_dims)
                },
            )
            for isim, sim in enumerate(sims)
        ]
        overlaps_sims = [sim.squeeze() for sim in overlaps_sims]
        return overlaps_sims

    def calc_metrics(self, results, labels):
        mappings0 = results['mappings']
        mappings = {labels[index]: mapping.data[0].tolist() for index, mapping in mappings0.items()}

        distances = [np.linalg.norm(param_utils.translation_from_affine(mapping.data[0]))
                     for mapping in mappings0.values()]
        if len(distances) > 2:
            # Coefficient of variation
            cvar = np.std(distances) / np.mean(distances)
            var = cvar
        else:
            size = get_sim_physical_size(results['sims'][0])
            norm_distance = np.sum(distances) / np.linalg.norm(size)
            var = norm_distance

        residual_errors = {labels[key[0]] + ' - ' + labels[key[1]]: value
                           for key, value in results['residual_errors'].items()}
        if len(residual_errors) > 0:
            residual_error = np.nanmean(list(residual_errors.values()))
        else:
            residual_error = 1

        registration_qualities = {labels[key[0]] + ' - ' + labels[key[1]]: value.item()
                                  for key, value in results['registration_qualities'].items()}
        if len(registration_qualities) > 0:
            registration_quality = np.nanmean(list(registration_qualities.values()))
        else:
            registration_quality = 0

        #overlap_metrics = self.calc_overlap_metrics(results)

        #nccs = {labels[key[0]] + ' - ' + labels[key[1]]: value
        #         for key, value in overlap_metrics['ncc'].items()}
        #ncc = np.nanmean(list(nccs.values()))

        #ssims = {labels[key[0]] + ' - ' + labels[key[1]]: value
        #         for key, value in overlap_metrics['ssim'].items()}
        #ssim = np.nanmean(list(ssims.values()))

        summary = (f'Residual error: {residual_error:.3f}'
                   f' Registration quality: {registration_quality:.3f}'
        #           f' NCC: {ncc:.3f}'
        #           f' SSIM: {ssim:.3f}'
                   f' Variation: {var:.3f}')

        return {'mappings': mappings,
                'variation': var,
                'residual_error': residual_error,
                'residual_errors': residual_errors,
                'registration_quality': registration_quality,
                'registration_qualities': registration_qualities,
         #       'ncc': ncc,
         #       'nccs': nccs,
         #       'ssim': ssim,
         #       'ssims': ssims,
                'summary': summary}

    def save_video(self, output, sims, fused_image):
        logging.info('Creating transition video...')
        pixel_size = [si_utils.get_spacing_from_sim(sims[0]).get(dim, 1) for dim in 'xy']
        params = self.params
        nframes = params.get('frames', 1)
        spacing = params.get('spacing', [1.1, 1])
        scale = params.get('scale', 1)
        transition_filename = output + 'transition'
        video = Video(transition_filename + '.mp4', fps=params.get('fps', 1))
        positions0 = np.array([si_utils.get_origin_from_sim(sim, asarray=True) for sim in sims])
        center = np.mean(positions0, 0)
        window = get_image_window(fused_image)

        max_size = None
        acum = 0
        for framei in range(nframes):
            c = (1 - np.cos(framei / (nframes - 1) * 2 * math.pi)) / 2
            acum += c / (nframes / 2)
            spacing1 = spacing[0] + (spacing[1] - spacing[0]) * acum
            for sim, position0 in zip(sims, positions0):
                transform = param_utils.identity_transform(ndim=2, t_coords=[0])
                transform[0][:2, 2] += (position0 - center) * spacing1
                si_utils.set_sim_affine(sim, transform, transform_key=self.transition_transform_key)
            frame = fusion.fuse(sims, transform_key=self.transition_transform_key).squeeze()
            frame = float2int_image(normalise_values(frame, window[0], window[1]))
            frame = cv.resize(np.asarray(frame), None, fx=scale, fy=scale)
            if max_size is None:
                max_size = frame.shape[1], frame.shape[0]
                video.size = max_size
            frame = image_reshape(frame, max_size)
            save_tiff(transition_filename + f'{framei:04d}.tiff', frame, None, pixel_size)
            video.write(frame)

        video.close()
