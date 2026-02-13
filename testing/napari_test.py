from qtpy.QtCore import QObject, QThread, Signal, Slot
from threading import Thread


class NapariTest(QThread):
    update_napari_signal = Signal(str, list, list)

    def __init__(self):
        super().__init__()
        import napari
        self.viewer = napari.Viewer()
        self.update_napari_signal.connect(self.update_napari)

    def run(self):
        self.update_napari_signal.emit('test_layer', [[0, 0], [0, 1], [1, 1], [1, 0]], [0])

    @Slot(str, list, list)
    def update_napari(self, layer_name, shapes, labels):
        if len(shapes) > 0:
            text = {'string': '{labels}'}
            features = {'labels': labels}
            self.viewer.add_shapes(shapes, name=layer_name, text=text, features=features, opacity=0.5)
            self.viewer.show()

if __name__ == '__main__':
    import napari
    napari_test = NapariTest()
    napari_test.start()
    napari.run()
