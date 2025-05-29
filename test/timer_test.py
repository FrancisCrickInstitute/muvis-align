from src.Timer import Timer

if __name__ == "__main__":
    with Timer('test', verbose=True):
        for i in range(1000000):
            pass
