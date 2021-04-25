# BuildingAI-project
Building AI course project of Frank

# Project Title

Final project for the Building AI course: Applying AI methods on Shipment Event Log Datat

## Summary

In Freight Forwarding a shipment is described by a data record defining the sending and receiving parties, the details transport legs  (locations and involved parties like carriers) and the details of goods.
In modern transport management systems during the different phases of the transport a lot of event data is generated wihich is then logged in an event log file in general.
This event log data is "very untidy" and not well structured 
1) since the sources of event data are not always reliable, 
2) manual handling of a shipment highly depends on the freight forwarder dealing with the shipment
3) there are many external conditions impacting the way the shipment can travel from A to B (wheather conditions, peak and off-peak effects, all kinds of force majeur.

So though there is in general a quite large "data pool" it is very hard to draw the right conclsion from the data like
- deriving the optimal process flow for certain kinds of shipments
- recognizing anomalies
- predict expected volume of shipments on a certain trade lane
- predict travelling duration under certain conditions.

This project shall examine the possibilities to apply methods of AI to the event log data in order to find out if this could help to make use of the event log data for process mining and predition of volume (for capacity planing), of travel duration and of potential problems (both for better clients' supply chain planing).

## Background

This project aims at solving the limitations to read  conclusions from very unstructured shipment event log data which is collected from different data sources and the shipment-individual data quality is very limited.
This is an experimental project to apply methods of AI to this data pool.

The problem is a very common problem - for example classical process analysis methods show that within a sample of 50 pre-selected and pre-classified shipment the event log show 50 variants of event flows which means 50 variants of process flows. This holds for all sample pre-selected shipments types.

I am pretty sure that this statement is valid for most  freight forwarders since they work under similar conditions using same data sources like carrier data and tracking data sources.

Personally I see big potential to use event log data for process optimization, productivity increase and predictions of volumes and problems and thereby improve supply chain sustainability and efficiency. With all the trade and transport streams around the world there shall be high motivation to optimize supply chain processes to safe on CO2 and reduce the impact on the climate.

These are the steps and partial problems solutions I plan to go for:

* problem 1: Cleaning and structuring shipment event log data: finding event pattern on the shipments
* problem 2: classifying shipments by event data like "normal" or "abnormal" event log pattern = process flow, 1-leg or 2-leg or 3-leg main transport leg
* problem 3: deriving standard workflow templates from classification, defining a optimal process handling on the freight forwarders side
* problem 4: try to apply AI methods like Next neighbour, linear or logistic regression, neuronal networks to predict shipment process characteristics like duration and aggregated volumes and problems. 


This topic of productivity and optimization lies within the area of my job tasks.


## How is it used?

The solution could be used by process engineers in the transport industry to define Standard Operationd Procedures (SOPs) based on realistic event data.
Subsequently freight forwarders could work along a workflow templates which is based on common pattern if the shipment is "common" or on specialized workflow templates if the shipment conditions.

Freight forwarders might get predictions of anomalies and take couter measures or at least give warnings in case of sensitive supply chain processes impacted.

Capacity planners could use it for more precise and reliable freight space allocations.

Sales managers could use it to for rate predictions and tailored offers.

Of course it depends on "data scientists" extracting event log data and enhancing the data quality by the methods found in this project and providing the results so that the above described conclusions can be drawn.  

All the above describe usages contribute to making transport and supply chain processes more efficient avoiding unexpected impacts, better use of capacity and less "expensive" and less eficient / sustainable counter-measures in case of anomalities.




![Container ship](https://upload.wikimedia.org/wikipedia/commons/2/20/Container_ship_Yorktown_Express_%282%29.jpg)



This is how you create code examples:
```
import math
import random
import numpy as np
import io
from io import StringIO

#DATA Block

text = '''Humpty Dumpty sat on a wall
Humpty Dumpty had a great fall
all the king's horses and all the king's men
couldn't put Humpty together again'''

def distance(row1, row2):
    # replace the following by a function that returns the sum of differences between the
    # words in row1 and row2. this is the Manhattan distance.
    # you can assume that row1 and row2 are list with equal length, containing numeric values.
    dist_h = 0
    for i in range (len(row1)):
        dist_h += abs(row1[i]-row2[i])
    return dist_h

def all_pairs(data,N):
    # this calculates the distances between all sentence pairs in the data
    # you do not need to change this
    i=0
    j=0
    dist_arr = np.empty((N, N), dtype=np.float)
    for i in range (0, N):
        for j in range (0, N):
           if j == i:
                dist_arr[i][j]=np.inf
           else:
                dist_arr[i][j]= distance (data[i], data[j])
    #print(dist_arr)
    return dist_arr

def find_nearest_pair(data):
    N = len(data)  
    dist = np.empty((N, N), dtype=np.float)
    dist = np.array(all_pairs(data,N))
    #print(dist)   
    print(np.unravel_index(np.argmin(dist), dist.shape))



def main(text):
    # tasks your code should perform:

    # 1. split the text into words, and get a list of unique words that appear in it
    # a short one-liner to separate the text into sentences (with words lower-cased to make words equal 
    # despite casing) can be done with 
    # docs = [line.lower().split() for line in text.split('\n')]
    docs = [line.lower().split() for line in text.split('\n')]
    #print (docs)
    N = len(docs)
    #print (N)
    # create the vocabulary: the list of words that appear at least once
    vocabulary = list(set(text.lower().split()))
    #print (vocabulary)
    
    # 2. go over each unique word and calculate its term frequency, and its document frequency
    df = {}
    tf = {}  

    for word in vocabulary:
        # tf: number of occurrences of word w in document divided by document length
        # note: tf[word] will be a list containing the tf of each word for each document
        # for example tf['he'][0] contains the term frequence of the word 'he' in the first
        # document
        tf[word] = [doc.count(word)/len(doc) for doc in docs]

        # df: number of documents containing word w
        df[word] = sum([word in doc for doc in docs])/N
    #print (tf)
    #print (df)   

    # 3. after you have your term frequencies and document frequencies, go over each line in the text and 
    # calculate its TF-IDF representation, which will be a vector
    tfidf_array = []
    for doc_index, doc in enumerate(docs):
        tfidf = []
        #print (doc_index)
        for word in vocabulary:
            # ADD THE CORRECT FORMULA HERE. Remember to use the base 10 logarithm: math.log(x, 10)
            #print (word)
            tfidf.append(tf[word][doc_index]*math.log(1/df[word],10)) 

        #print(tfidf)  
        tfidf_array.append(tfidf)
    #print ('--------------------------------------------')
    #print (tfidf_array)

    # 4. after you have calculated the TF-IDF representations for each line in the text, you need to
    # calculate the distances between each line to find which are the closest.

    find_nearest_pair(tfidf_array)
    print (tfidf_array)


main(text)


```


## Data sources and AI methods

Data comes from the event log of the transport management system, CargoWise One by WiseTech Global Ltd., of my company SENATOR INTERNATONAL.
Due to confidentiality data is anonymized, real IDs are replaced and event details are generalized.
In the first phase tere is only a reduced subset of event data to work-out the methods of data cleansing and sorting and data analytics.

I have collected manually 19 shipment event log files and stripped them to the necessary data used here for a special trade lane between China (CN) and ermany (DE). 

The data is provided in an ecxel file of following sturcture

| Shipment ID | EVENT Descr. | Event time stamp 
| ----------- | ------------ | ----------------
| ID1         | Event 1(ID1) | Time 1 (ID1)
| ...         | ...          | ...
| ID1         | Event n(ID1) | Time n (ID1)
| ID2         | Event 1(ID2) | Time 1 (ID2)
| ...         | ...          | ...
| ID2         | Event m(ID2) | Time m (ID2)
| ...         | ...          | ...
| ...         | ...          | ...
| IDk         | Event 1(IDk) | Time 1 (IDk)
| ...         | ...          | ...
| IDk         | Event p(IDk) | Time p (IDk)


First step will be to find out for all sample file which are the most common events, how often do they appear and which are single events or repeating events. Is there a common or most likely order of events.

## First solution approach / Code example

While working with the Event Data Logs I found the analogy to working with text in the course Building AI: https://buildingai.elementsofai.com/Machine-Learning/working-with-text:
I have transformed for each shipment the event into the list of events like a list of words in each sentence and applied the "bag of words" / "Term Frequency Inverse Document Frequency (tf-idf)" method.

I have used the code template from the course.



## Challenges

This project is a learning project for me - I have not put  much effort in finding out whether there are already established tools like neuronal network frame works where I could just feed in the data into but the main challenge at the moment is to do the best and reasonable data preparation.
For my company this approach is - as far as I know - new. It was not looked at event data in this way yet.
Of course it would be possible to involve  AI experts and consultants to solve the topics in question, but everybody knows the limitations the "Pros and Cons" of external solutions.

Then an experimental and interative process will start to find the appropriate AI method for the different tasks  like preditions, process mining and anomaly recognition.
The outcome is not yet determined.

The ethical impact of this approach is positive: If the expected goal is reached there would be an increase in sustainability of supply chains. Of course all increases of productivity might impact the situation of employment (less jobs, more stress on employees, more stupid routine work left) but currently it is very hard to find enough staff for this kind of jobs, so a relief by reducing inefficient processes  could not harm.

## What next?

Starting the journey to apply different methods of AI to find out their suitability for solving the above mentioned problems  and reaching the corresponding goals.

## Acknowledgments

* list here the sources of inspiration 
* do not use code, images, data etc. from others without permission
* when you have permission to use other people's materials, always mention the original creator and the open source / Creative Commons licence they've used
  <br>For example: [Tvabutzku1234, Public domain, via Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Container_ship_Yorktown_Express_(2).jpg#filelinks) / [CC BY 2.0](https://creativecommons.org/licenses/by/2.0)
* Code template from exercise 18 of course Building AI, exercise to train "Term Frequency Inverse Document Frequency (tf-idf)"
* https://buildingai.elementsofai.com/Machine-Learning/working-with-text
