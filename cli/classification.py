import os,sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.remove(os.path.dirname(__file__))

from classification.classifier import file_image_classifier_factory

from cli.prediction import PredictionCLI

def cli_main():
    PredictionCLI(file_image_classifier_factory).run()

"""
def cli_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--mode", action=enum_action(ClassificationMode), help="The classification mode to use")
    parser.add_argument("--min-confidence", type=int, choices=range(0,100), help="Minimum confidence to find")
    parser.add_argument("-c", "--class", type=str, nargs='*', help="class(es) to find", dest="classes")
    parser.add_argument("images", type=str, nargs='+', help="The image file(s) to process")

    args = parser.parse_args()

    results = classify_images(
        images=args.images,
        mode=args.mode,
        min_confidence=args.min_confidence,
        classes=args.classes,
    )

    for result in results:
        print(result)
"""

if __name__ == "__main__":
    cli_main()