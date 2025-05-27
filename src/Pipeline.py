import datetime
from distributed import Client, performance_report
import logging
import numpy as np
import os
import pandas as pd
from threading import Thread
from tqdm import tqdm

from src.constants import version
from src.image.source_helper import get_images_metadata
from src.util import dir_regex, get_filetitle, find_all_numbers, split_numeric_dict


class Pipeline(Thread):
    def __init__(self, params, viewer=None):
        super().__init__()
        self.params = params
        self.viewer = viewer

        self.params_general = params['general']
        self.init_logging()

        napari_ui = 'napari' in self.params_general.get('ui', '')
        if napari_ui:
            from src.MVSRegistrationNapari import MVSRegistrationNapari
            self.mvs_registration = MVSRegistrationNapari(self.params_general, self.viewer)
        else:
            from src.MVSRegistration import MVSRegistration
            self.mvs_registration = MVSRegistration(self.params_general)

    def init_logging(self):
        self.verbose = self.params_general.get('verbose', False)
        self.verbose_mvs = self.params_general.get('verbose_mvs', False)
        self.log_filename = self.params_general.get('log_filename', 'logfile.log')
        log_format = self.params_general.get('log_format')
        basepath = os.path.dirname(self.log_filename)
        if not os.path.exists(basepath):
            os.makedirs(basepath)

        handlers = [logging.FileHandler(self.log_filename, encoding='utf-8')]
        if self.verbose:
            handlers += [logging.StreamHandler()]

        logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers, encoding='utf-8')

        # verbose external modules
        if self.verbose_mvs:
            # expose multiview_stitcher.registration logger and make more verbose
            mvsr_logger = logging.getLogger('multiview_stitcher.registration')
            mvsr_logger.setLevel(logging.INFO)
            if len(mvsr_logger.handlers) == 0:
                mvsr_logger.addHandler(logging.StreamHandler())
        else:
            # reduce verbose level
            for module in ['multiview_stitcher', 'multiview_stitcher.registration', 'multiview_stitcher.fusion']:
                logging.getLogger(module).setLevel(logging.WARNING)

        for module in ['ome_zarr']:
            logging.getLogger(module).setLevel(logging.WARNING)

        logging.info(f'muvis-align version {version}')

    def run(self):
        break_on_error = self.params_general.get('break_on_error', False)

        with Client(processes=False) as client:
            if self.verbose:
                print(client)

            for operation_params in tqdm(self.params['operations']):
                error = False
                timestamp_filename = (os.path.splitext(self.log_filename)[0]
                                      + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
                input_path = operation_params['input']
                logging.info(f'Input: {input_path}')
                with performance_report(filename=timestamp_filename + "_report.html"):

                    try:
                        self.run_operation(operation_params)
                    except Exception as e:
                        logging.exception(f'Error processing: {input_path}')
                        print(f'Error processing: {input_path}: {e}')
                        error = True

                if error and break_on_error:
                    break

        logging.info('Done!')

    def run_operation(self, params):
        operation = params['operation']
        use_global_metadata = 'global' in params.get('source_metadata', '')
        metadata_summary = self.params_general.get('metadata_summary', False)

        filenames = dir_regex(params['input'])
        filenames = sorted(filenames, key=lambda file: list(find_all_numbers(file)))    # sort first key first
        if len(filenames) == 0:
            logging.warning(f'Skipping operation {operation} (no files)')
            return False
        elif self.verbose:
            logging.info(f'# total files: {len(filenames)}')

        operation_parts = operation.split()
        if 'match' in operation_parts:
            # check if match label provided
            index = operation_parts.index('match') + 1
            if index < len(operation_parts):
                match_label = operation_parts[index]
            else:
                match_label = None
            matches = {}
            for filename in filenames:
                parts = split_numeric_dict(filename)
                match_value = parts.get(match_label)
                if match_value is not None:
                    if match_value.isdecimal():
                        match_value = int(match_value)
                    if match_value not in matches:
                        matches[match_value] = []
                    matches[match_value].append(filename)
                if len(matches) == 0:
                    matches[0] = filenames
            filesets = []
            fileset_labels = []
            for label in sorted(matches):
                filesets.append(matches[label])
                fileset_labels.append(f'{match_label}:{label}')
            logging.info(f'# matched file sets: {len(filesets)}')
        else:
            filesets = [filenames]
            fileset_labels = [get_filetitle(filename) for filename in filenames]

        metadatas = []
        rotations = []
        global_center = None
        if metadata_summary or use_global_metadata:
            for fileset, fileset_label in zip(filesets, fileset_labels):
                metadata = get_images_metadata(fileset)
                if metadata_summary:
                    logging.info(f'File set: {fileset_label} metadata:\n' + metadata['summary'])
                metadatas.append(metadata)
            if use_global_metadata:
                global_center = np.mean([metadata['center'] for metadata in metadatas], 0)
                rotations = [metadata['rotation'] for metadata in metadatas]
                # fix missing rotation values
                rotations = pd.Series(rotations).interpolate(limit_direction='both').to_numpy()

        ok = False
        for index, (fileset, fileset_label) in enumerate(zip(filesets, fileset_labels)):
            if len(filesets) > 1:
                logging.info(f'File set: {fileset_label}')
            center = global_center if use_global_metadata else None
            rotation = rotations[index] if use_global_metadata else None
            ok |= self.mvs_registration.run_operation(fileset_label, fileset, params,
                                                      global_center=center, global_rotation=rotation)

        return ok
