from ..classification.classifier import image_classifier_factory

from .prediction import FactoryImagePredictionCLI


def cli_main():
    FactoryImagePredictionCLI(image_classifier_factory).run()


if __name__ == "__main__":
    cli_main()
