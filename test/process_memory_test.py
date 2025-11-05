import yaml

from src.MVSRegistration import MVSRegistration


def test_process_memory(params, filenames):
    with open(params, 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    reg = MVSRegistration(params['general'])
    reg.run_operation('1', filenames, params['operations'][0])


if __name__ == '__main__':
    params = 'resources/params_test.yml'
    filenames = ['D:/slides/13457227.zarr', 'D:/slides/13457227.zarr']
    test_process_memory(params, filenames)
