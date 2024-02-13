from ..detection.detector import object_detector_factory

from .prediction import FactoryImagePredictionCLI

def cli_main():
    FactoryImagePredictionCLI(object_detector_factory).run()

if __name__ == "__main__":
    cli_main()