import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import *
from tkinter import messagebox


class View:

    def __init__(self):
        self.create_gui()

    def create_gui(self):
        """
        Creates the tkinter GUI.
        """
        self.window = tk.Tk()
        self.window.title("Sentiment Classifier")
        self.window.geometry('500x200')

        self.lbl = Label(self.window, text="Enter Sentence")
        self.lbl.place(x=20, y=10)

        self.sentence = StringVar()
        self.sentence = Entry(self.window, width=50, textvariable=self.sentence)
        self.sentence.place(x=120, y=10)
        self.emlbl = Label(self.window, text="Choose the word embedding model")
        self.emlbl.place(x=20, y=40)

        self.c = StringVar(self.window, "1")
        self.check1 = Radiobutton(self.window, text="Google news", variable=self.c, value="w2gn_model")
        self.check1.place(x=250, y=40)
        print(self.c)
        self.check2 = Radiobutton(self.window, text="SO", variable=self.c, value="w2so_model")
        self.check2.place(x=350, y=40)
        print(self.c)

        self.lbl2 = Label(self.window, text="Sentiment:")
        self.lbl2.place(x=20, y=130)

        self.ans_lbl = Label(self.window)
        self.ans_lbl.place(x=120, y=130)

        self.classlbl = Label(self.window, text="Choose the classifier")
        self.classlbl.place(x=20, y=65)

        self.classifier = StringVar(self.window, "1")
        self.R1 = Radiobutton(self.window, text="LSTM", padx=20, variable=self.classifier, value="LSTM")
        self.R1.pack(anchor=W, side=LEFT, ipadx=10)
        print(self.classifier)

        self.R2 = Radiobutton(self.window, text="BILSTM", padx=20, variable=self.classifier, value="BILSTM")
        self.R2.pack(anchor=W, side=LEFT, ipadx=10)
        print(self.classifier)

        self.R3 = Radiobutton(self.window, text="CNN", padx=20, variable=self.classifier, value="CNN")
        self.R3.pack(anchor=W, side=LEFT, ipadx=10)
        print(self.classifier)

        self.menubar = Menu(self.window, tearoff=5)
        self.window.config(menu=self.menubar)
        self.helping = Menu(self.menubar, tearoff=0)
        self.helping.add_command(label="About", command=self.popup_window_help)
        self.menubar.add_cascade(label="Help", menu=self.helping)

        # self.button_classify = tk.Button(window, text ="classify", command = self.classify_sentence_google)
        self.button_classify = tk.Button(self.window, text="classify")
        self.button_classify.place(x=370, y=90)

    def popup_window_help(self):
        """
        Displays a Help window, with a help message.
        """
        self.window = tk.Toplevel()

        self.label = tk.Label(self.window,
                              text='''The application identifies the sentiment of an input sentence. \n Type a sentence, use the radio buttons to make selections. \n Click the classification button to see the sentiment''')
        self.label.pack(fill='x', padx=50, pady=5)

    def error_popup_selections(self):
        """
        Displays a warning message box urging the user to select the Word2Vec model and deep-learning classifier.
        """
        messagebox.showwarning("Warning", "Please select word embedding model and classifier")

    def error_popup_classifier(self):
        """
        Displays a warning message box urging the user to select a deep learning classifier.
        """
        messagebox.showwarning("Warning", "Please select a classifier")

    def error_popup_embedding(self):
        """
        Displays a warning message box urging the user to select a Word2Vec model.
        """
        messagebox.showwarning("Warning", "Please select a word embedding model")

    def error_popup(self):
        """
        Displays a warning message when there is an unexpected error.
        """
        messagebox.showwarning("Warning", "ERROR")

    def error_popup_sentence(self):
        """
        Displays a warning message box urging the user to input a sentence.
        """
        messagebox.showwarning("Warning", "Enter Sentence")
