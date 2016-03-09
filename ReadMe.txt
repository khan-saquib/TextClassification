1. Please make sure that the StopWords.txt file is in the same folder as the NNWeka.java file.
2. Keep weka.jar in the same folder as the NNWeka.java file.
3. This program gets the training data during every execution. Hence, you will have to specify the dataset folder as the first argument.
4. I have implemented two algorithms. Naive Bayes and Logistic Regression. Filtering attributes + Logistic Regression takes almost an` hour to execute. Hence, I have commented them. If you want to run Logistic Regression, do the following steps: 

	a. Comment code from line 303 to 309. 
	b. Uncomment code from line 314 to 329 and 285 to 299. 
	c. Run the code again for output of Logistic Regression.


Example: 
My folder hierarchy is:
	C:\Users\Saquib\Desktop
	--project
	----dev
	----train
	----class_name.txt
	----dev_label.txt
My run command for given dataset is:
java -cp weka.jar:. NNWeka "C:\Users\Saquib\Desktop\project" "C:\Users\Saquib\Desktop\project\dev" "output.txt"




Generic Compile Command: 
javac -cp weka.jar NNWeka.java

Generic Run Command: 
java -cp weka.jar:. NNWeka <folder_path_of_dataset> <folder_path_of_testing_data> <PATH+FILENAME_OF_OUTPUT_FILE>
