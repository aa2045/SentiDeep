from gui_model import Model
from gui_view import View


class Controller:
    """Calls the business logic from Model class, to identify the sentiment of sentence and updates View."""
    def __init__(self):
        """
        Attributes:
            view  (View):
            model (Model):


        """
        self.view = View()
        self.model = Model()
        print("i am in controller init")

    def run(self):
        """
        Runs tkinter event loop, by checking for button clicks.
        """
        self.view.button_classify.bind("<Button-1>", self.classify_sentence)
        self.view.window.mainloop()
        print("i am in controller run")

    def classify_sentence(self, event):
        """
        Classifies the sentiment of sentence, by calling method of Model and updates the View accordingly.
        Args:
            event:

        """
        try:
            if (len(self.view.sentence.get()) == 0):
                self.view.error_popup_sentence()
                print("i am in controller try-if")
            else:
                print("can i ever go to try-else??")
                polarity = self.model.sentiment_sentence(self.view.sentence.get(), self.view.classifier.get(),
                                                          self.view.c.get())
                print(self.view.classifier.get())
                print(self.view.c.get())
                self.view.ans_lbl.config(text=polarity)
                return polarity
        except:
            if (len(self.view.sentence.get()) == 0):
                self.view.error_popup_sentence()

            elif ((self.view.c.get() != "w2gn_model" and self.view.c.get() != "w2so_model") and (
                    self.view.classifier.get() != "LSTM" and self.view.classifier.get() != "BILSTM" and self.view.classifier.get() != "CNN")):
                self.view.error_popup_selections()

            elif (
                    self.view.classifier.get() != "LSTM" and self.view.classifier.get() != "BILSTM" and self.view.classifier.get() != "CNN"):
                self.view.error_popup_classifier()
            elif (self.view.c.get() != "w2gn_model" and self.view.c.get() != "w2so_model"):
                self.view.error_popup_embedding()



            else:
                self.view.error_popup()