# Word suggestion using LSTM Recurrent Neural Network with TensorFlow
This code creates a simple system able to suggest the next word based on the previous words provided. It can also be used to create a specific text based on some other texts.<BR><BR>
The system uses a LSTM (long short term) Recurrent Neural Network model to learn from a dataset with 100 questions extracted from a help desk system.
  
### How to run the code
To prepare your data, ensure the dataset is on the same directory as your code and run *prep_data.py*. The prepared dataset and other supporting files will be generated.<BR>
To train the neural network, uncomment lines 96 and 97 from the *get_word.py* file and run it.<BR>
To use the neural network also run the *get_word.py* file calling the *get_word* function. Unless you need to retrain your network you can keep the *training* function commented out.

### Files description
* **prep_data.py** - generates the dictionary of words based on the dataset and also creates a pickle file where each phrase is a list object and each item of the list is word.
* **lstm_sentence_nn.py** - creates the recurrent neural network model and perform all the network training process saving the trained model at the end of the process.
* **get_word.py** - where you can use the model, provides the *get_word* function which creates a simple user experience in the console so you can test and play around with the trained model.
