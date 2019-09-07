# Assignment-16
Using TFRecords to load the data and achieved an accuracy of above 85% test data of 10,000 samples, the model is trained with 40,000 samples of cifar10 dataset.

How to use the repo:
1. Just open the jupyter notebook in colab and load the python files from the same repo.
2. First run the file "TFRecords.py", which creates binary proto buff files in the TFRecords format.
3. Once the TFRecords files are generated for train, test and valuation data are created. Run the file "read_TFRecords.py" file which reads the data from TFRecords files and trains and evaluates the model.
Step 2 and 3 are done if the jupyter notebook is executed.


In the upcoming Assignments, I will update the input data pipeline with efficinet and fast loading and with better understanding of how the pipeline works.
