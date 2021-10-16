from cx_Freeze import setup, Executable
setup(name = "SentimentAnalysis" ,
      version = "1.0" ,
      description = "Sentiment Analysis to identify developers' sentiments" ,
      executables = [Executable("gui_main.py")])