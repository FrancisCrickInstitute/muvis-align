from qtpy.QtCore import QObject, Signal, Slot

from src.MVSRegistration import MVSRegistration


class MVSRegistrationNapari(MVSRegistration, QObject):
    update_napari_signal = Signal(str, list, list)

    def __init__(self, params_general, viewer):
        super().__init__(params_general)
        self.viewer = viewer
        self.update_napari_signal.connect(self.update_napari)

    @Slot(str, list, list)
    def update_napari(self, layer_name, shapes, labels):
        if len(shapes) > 0:
            text = {'string': '{labels}'}
            features = {'labels': labels}
            self.viewer.add_shapes(shapes, name=layer_name, text=text, features=features, opacity=0.5)
            self.viewer.show()
