# SentiDeep
Sentiment analysis using deep learning stacked layers of LSTM, BILSTM, CNN

Sentiment Analysis for Software Engineering text

The src folder contains the .py files to perform sentiment analysis using deep-learning models, such as LSTM, BILSTM and CNN.

To train the models, CLI application is developed, where user inputs the filename, word embedding model name, number of epochs and classifier model.
File to run- cli_main.py

GUI application is also developed and a standalone is available where user can input the sentence, select the classifier model and word embedding model.
sample_texts.csv can be used to pick sentences and input to the classifiers in the GUI.

Software specific word embedding model is developed by using the 2 million StackOverflow posts to create the word embedding model, run create_embeds.py
The StackOverflow datadump is available at: https://heriotwatt-my.sharepoint.com/:f:/g/personal/aa2045_hw_ac_uk/EjR_wWIyOjZNjz9njobR3ZkByobsKdue0zf14eb4-GhPrw?e=SvQteo
Place the datadump in the src folder and run the create_embeds.py to create software embeddings.

Google news word embedding is found in https://heriotwatt-my.sharepoint.com/:f:/r/personal/aa2045_hw_ac_uk/Documents/mpcode/src?csf=1&web=1&e=I4AC71
Sphinx documentation for all the code files is saved in the documentation directory.
