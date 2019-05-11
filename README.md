# Project-Group-2


# Log Mining using Natural Language Processing for Anomaly Detection



### **Team Members:**

- Ankita Chikodi                              
- Arkil Thakkar                                 
- Nehal Sharma                         
- Shravani Pande



### **Project Description:**

In our project, we aim to reduce human intervention for log file processing by proposing a novel approach that considers logs as regular text (as opposed to related works that exploit the little structure imposed by log formatting). Our methodology makes use of modern algorithms from natural language processing, Kafka and Spark. The resulting pipeline is generic, computationally efficient, and requires minimal intervention. Our approach demonstrates a strong predictive performance (≈ 95% accuracy) using Random forest Classifier.

### **System Architecture:**

The log data is transferred through Kafka producer which is received as a Spark streaming object. The log data in the Spark streaming object is pre-processed and applied to the pre-trained model which is build using the older logs. After pre-processing, the model will predict whether it is an anomaly and then it will send a mail to the user.

![image](https://user-images.githubusercontent.com/47070167/57565603-a8b03980-7375-11e9-9c41-86370a0de1ad.png)



### **Data Preprocessing:**

**Raw Data**
Log data generated through HDFS 
![image](https://user-images.githubusercontent.com/47070167/57565520-5884a780-7374-11e9-9dfe-9dd4e631451f.png)


**Log DataFrame**
Unstructured logs preprocessed into dataframe
![image](https://user-images.githubusercontent.com/47070167/57565523-74884900-7374-11e9-8311-9aaaf5b3e4f7.png)


**Event Sequence**
BlockID having different events 
![image](https://user-images.githubusercontent.com/47070167/57565521-6a664a80-7374-11e9-8b18-ea7ab5d0cd22.png)


**Label Dictionary**
Dictionary of blockIDs and event sequences
![image](https://user-images.githubusercontent.com/47070167/57565539-aef1e600-7374-11e9-9a77-cb7cd0b8c5a2.png)


**Label Mapping**
Mapping blockIDs with associated label i.e. Anomaly or Normal
![image](https://user-images.githubusercontent.com/47070167/57565534-a00b3380-7374-11e9-8da2-51ae87e11721.png)

### **Natural Language Processing:**
**TF-IDF**
Building matrix by converting event sequences into TF-IDF form
![image](https://user-images.githubusercontent.com/47070167/57565552-ed87a080-7374-11e9-92ae-eb77ee142aa4.png)

**Normalized Vector**
Normalizes the matrix generated from TF-IDF
![image](https://user-images.githubusercontent.com/47070167/57565561-0f812300-7375-11e9-8023-1360253fa09c.png)


### **Model Building:**
**Dumping Model**
Storing the trained model
![image](https://user-images.githubusercontent.com/47070167/57565565-2758a700-7375-11e9-991f-48f016dbf913.png)


### **Data Pipeline:**
**Kafka Logs**


![image](https://user-images.githubusercontent.com/47070167/57565902-908ee900-737a-11e9-8d58-2343fba31175.png)

**Output:**
![image](https://user-images.githubusercontent.com/47070167/57566127-c7b2c980-737d-11e9-86ed-0b1e935b19a4.png)


### **How to use this:**

- Start Zookeeper
   zkserver
- Start Kafka Server
  .\bin\windows\kafka-server-start.bat .\config\server.properties 
- Create a Kafka topic
  kafka-topics.bat --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic test
- Initialize a local producer 
  kafka-console-producer.bat --broker-list localhost:9092 --topic test \--new-producer < HDFS.log

- Spark 
  ./spark-submit.sh '--jars spark-streaming-kafka-0-8-assembly_2.11-2.3.3.jar pyspark-shell Spark_Log_mining.py



