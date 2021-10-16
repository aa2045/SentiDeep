import argparse

class View:
    """View class containing the code to specify the arguements for Command line interface

    """

        #self.create_gui()
    def create_cli(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("file", help="filename")
        parser.add_argument("wordembedding",
                            help="wordembedding model, enter \"SO\" "
                                 "for Software embedding and \"Google News\" "
                                 "for Google news embedding")
        parser.add_argument("epochs", help="epochs")
        parser.add_argument("model", help="choose the model you want to train",
                            choices=["CNN", "BILSTM", "LSTM"])

        # takes the arguments provided on the command line
        args = parser.parse_args()
        return args
