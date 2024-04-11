* Instructions to use NetworkController()
  - Any network that you want to use with the NetworkController should subclass the NetworkBase class defined in networkController.py
  - All functions defined within NetworkBase should be reimplemented inside your class with the same signatures (ie. same input and output)
  - After this all functionality of NetworkController() should be available to your model.
  - When writing tools for both the DeepSets model and the FakeTracks model, only assume that the methods defined in NetworkController exist. If you find you need another method, add this to NetworkController() and make sure that these methods are added to both models. 
The code is well documented, so check networkController for further details on what is available for use and how to use it. An example for using the NetworkController is given in example.py. 
