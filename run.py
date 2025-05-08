import argparse
import yaml

from src.Pipeline import Pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'multiview-stitcher')
    parser.add_argument('--params',
                        help='The parameters file',
                        default='resources/params.yml')

    args = parser.parse_args()
    print(f'Parameters file: {args.params}')
    with open(args.params, 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    napari_ui = 'napari' in params['general'].get('ui', '')
    if napari_ui:
        try:
            import napari
            viewer = napari.Viewer()
            pipeline = Pipeline(params, viewer)
            pipeline.start()    # run as thread
            napari.run()
        except ImportError:
            raise ImportError('Napari not installed.')
    else:
        pipeline = Pipeline(params)
        pipeline.run()
