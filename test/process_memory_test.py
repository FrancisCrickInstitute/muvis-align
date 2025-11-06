import yaml

from src.MVSRegistration import MVSRegistration


def test_process_memory(params, filenames):
    with open(params, 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    reg = MVSRegistration(params['general'])
    reg.run_operation('1', filenames, params['operations'][0])


if __name__ == '__main__':
    params = 'resources/params_test.yml'
    filenames = ['D:/slides/idr0026.zarr/0'] * 10
    test_process_memory(params, filenames)
