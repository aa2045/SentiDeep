from cli_model import ModelTrain
from cli_view import View


class Controller:
    """Calls the business logic from Model class, to train
    the specified sentiment classifier and display the classification results """

    def __init__(self):

        self.view = View()
        self.args = self.view.create_cli()
        self.model = ModelTrain(self.args.file)

    def run(self):
        """ Runs the program on CLI

        Returns:
            list of the evaluation metrics through all folds and prints the classification report.

        """
        result = None
        print(self.args.model)
        print(self.args.epochs)
        print(self.args.wordembedding)
        if self.args.model == "CNN":
            print("I am inside the if self.model == CNN")
            result = self.model.k_fold_train("CNN", self.args.wordembedding, self.args.epochs)
        elif self.args.model == "BILSTM":
            result = self.model.k_fold_train("BILSTM", self.args.wordembedding, self.args.epochs)
        elif self.args.model == "LSTM":
            result = self.model.k_fold_train("LSTM", self.args.wordembedding, self.args.epochs)
        else:
            print("Unsupported")
        print(result)
