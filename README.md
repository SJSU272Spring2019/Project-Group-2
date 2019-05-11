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



### **Data Preprocessing:**

**Raw Data**

![image](https://user-images.githubusercontent.com/47070167/57565520-5884a780-7374-11e9-9dfe-9dd4e631451f.png)

**Event Sequence**

![image](https://user-images.githubusercontent.com/47070167/57565521-6a664a80-7374-11e9-8b18-ea7ab5d0cd22.png)

**Log DataFrame**

![image](https://user-images.githubusercontent.com/47070167/57565523-74884900-7374-11e9-8311-9aaaf5b3e4f7.png)

**Label Mapping**

![image](https://user-images.githubusercontent.com/47070167/57565534-a00b3380-7374-11e9-8da2-51ae87e11721.png)

**Label Dict**

![image](https://user-images.githubusercontent.com/47070167/57565539-aef1e600-7374-11e9-9a77-cb7cd0b8c5a2.png)


**TF-IDF**

![image](https://user-images.githubusercontent.com/47070167/57565552-ed87a080-7374-11e9-92ae-eb77ee142aa4.png)

**Normalized Vector**

![image](https://user-images.githubusercontent.com/47070167/57565561-0f812300-7375-11e9-8023-1360253fa09c.png)

**Dumping Model**

![image](https://user-images.githubusercontent.com/47070167/57565565-2758a700-7375-11e9-991f-48f016dbf913.png)
