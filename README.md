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

This codes returns 
1) the number of shipments (rows in the event list)
2) the array of distances between the shipments using "Term Frequency Inverse Document Frequency (tf-idf)" method.
3) the pair of shipments with minimum distance.
```

import math
import random
import numpy as np
import io
from io import StringIO


#DATA Block: List of events (by Event ID), one shipment per line

events = '''E1 E2 E3 E4 E5 E5 E5 E6 E7 E8 E9 E10 E11 E12 E13 E14 E15 E16 E5 E5 E17 E18 E4 E19 E19 E20 E21 E22 E23 E17 E24 E25 E26 E27 E28 E26 E29 E30 E31 E32 E32 E33 E34 E35 E36 E37 E38 E39                        
E1 E2 E10 E5 E4 E13 E14 E16 E5 E5 E17 E3 E6 E7 E8 E9 E11 E12 E40 E14 E11 E13 E18 E4 E41 E42 E19 E19 E20 E43 E22 E23 E17 E24 E25 E26 E27 E26 E28 E31 E31 E32 E33 E34 E35 E37 E38 E39 E24                       
E1 E2 E4 E4 E44 E5 E6 E7 E8 E9 E45 E5 E10 E46 E46 E46 E16 E5 E5 E17 E17 E11 E12 E14 E15 E13 E14 E47 E18 E14 E48 E48 E48 E49 E46 E46 E14 E47 E50 E51 E24 E52 E3 E20 E43 E48 E48 E17 E25 E23 E22 E53 E14 E28 E31 E31 E32 E33 E33 E33 E34 E30 E35 E36 E29 E37 E54 E39 E26 E27 E55 E56
E1 E2 E4 E5 E6 E8 E45 E5 E10 E16 E5 E5 E17 E14 E13 E11 E14 E47 E18 E4 E14 E3 E17 E19 E19 E20 E43 E14 E47 E24 E52 E25 E23 E22 E53 E14 E28 E32 E31 E33 E34 E35 E26 E27 E26 E26 E55 E37 E38 E37 E29 E30 E36 E39                  
E2 E1 E4 E44 E4 E14 E5 E6 E7 E8 E9 E45 E14 E11 E12 E14 E13 E15 E14 E10 E16 E5 E18 E24 E18 E4 E14 E48 E48 E48 E48 E48 E48 E46 E46 E46 E52 E3 E49 E23 E22 E47 E50 E51 E46 E20 E43 E25 E17 E28 E30 E31 E32 E33 E34 E29 E26 E27 E35 E36 E37 E38 E39 E53        
E2 E3 E4 E5 E6 E7 E8 E9 E10 E14 E13 E11 E12 E15 E16 E5 E5 E17 E18 E4 E19 E19 E20 E43 E23 E22 E17 E24 E25 E26 E27 E28 E26 E29 E30 E31 E31 E32 E33 E34 E35 E36 E37 E38 E39                           
E1 E2 E4 E14 E5 E6 E7 E8 E9 E45 E14 E14 E13 E11 E12 E15 E14 E16 E5 E5 E10 E17 E18 E4 E14 E52 E22 E23 E3 E17 E47 E24 E24 E20 E43 E25 E24 E28 E30 E31 E32 E32 E33 E34 E35 E36 E29 E26 E27 E37 E38 E39 E53                   
E2 E1 E4 E4 E18 E4 E5 E14 E5 E10 E57 E14 E14 E13 E11 E16 E5 E18 E4 E14 E17 E52 E47 E14 E49 E3 E50 E58 E20 E43 E25 E22 E23 E28 E31 E31 E32 E33 E34 E35 E26 E27 E53                             
E1 E2 E4 E59 E6 E7 E8 E9 E5 E10 E16 E5 E5 E17 E11 E12 E15 E3 E60 E61 E62 E63 E14 E13 E64 E65 E66 E18 E4 E17 E19 E19 E20 E43 E25 E24 E23 E22 E26 E27 E28 E30 E31 E33 E31 E32 E33 E34 E31 E32 E33 E34 E35 E36 E38 E37 E39               
E2 E1 E56 E5 E6 E7 E8 E9 E67 E10 E12 E13 E15 E11 E16 E68 E5 E24 E24 E4 E17 E69 E70 E71 E72 E73 E73 E25 E23 E3 E22 E27 E28 E32 E31 E31 E33 E34 E35 E27 E29 E74 E30 E36 E37 E38 E75 E39                        
E1 E2 E4 E6 E59 E5 E8 E7 E9 E11 E10 E14 E13 E16 E5 E5 E17 E12 E15 E4 E76 E77 E78 E24 E79 E80 E81 E81 E25 E22 E3 E23 E27 E27 E29 E26 E58 E82 E20 E43 E28 E31 E32 E33 E34 E30 E35 E38 E37 E36 E39                     
E2 E1 E5 E4 E13 E14 E11 E10 E16 E5 E5 E17 E83 E42 E57 E18 E4 E14 E13 E23 E3 E14 E14 E43 E20 E84 E47 E14 E28 E22 E52 E28 E31 E31 E32 E33 E34 E58 E85 E35 E26 E27                              
E1 E10 E4 E2 E5 E11 E16 E5 E5 E17 E17 E14 E13 E47 E14 E4 E24 E47 E68 E47 E17 E14 E81 E52 E3 E58 E20 E43 E86 E86 E87 E25 E53 E23 E22 E14 E28 E32 E35 E88 E31 E33 E34 E82 E26 E27 E55                         
E1 E2 E10 E4 E5 E11 E16 E5 E5 E17 E14 E13 E48 E4 E47 E3 E14 E47 E20 E43 E89 E47 E14 E23 E90 E17 E84 E58 E52 E53 E91 E14 E31 E31 E33 E34 E26 E55                                  
E92 E1 E93 E2 E5 E5 E94 E95 E96 E96 E97 E98 E98 E96 E96 E16 E5 E11 E14 E13 E5 E47 E96 E14 E47 E96 E47 E96 E96 E99 E98 E98 E24 E24 E4 E100 E14 E52 E101 E58 E96 E102 E103 E104 E96 E53 E96 E105 E96 E14 E28 E26 E27 E106 E35 E107 E55 E82 E108 E109            
E1 E2 E59 E6 E7 E4 E5 E8 E9 E10 E11 E12 E40 E13 E14 E42 E16 E5 E5 E83 E17 E17 E11 E14 E13 E18 E4 E14 E3 E19 E19 E20 E43 E17 E110 E28 E24 E71 E72 E111 E72 E25 E22 E23 E53 E53 E69 E112 E26 E27 E26 E28 E31 E32 E33 E33 E34 E29 E30 E35 E113 E36 E37 E38 E39       
E2 E114 E93 E92 E1 E5 E5 E94 E95 E6 E96 E67 E8 E7 E9 E97 E98 E98 E96 E96 E11 E12 E15 E16 E5 E14 E13 E5 E47 E96 E14 E96 E96 E96 E99 E98 E98 E24 E24 E4 E58 E100 E14 E4 E52 E101 E102 E103 E104 E96 E96 E53 E14 E96 E105 E29 E115 E28 E30 E26 E27 E35 E116 E107 E106 E36 E55 E38 E37 E39 E82 E108
E1 E2 E4 E10 E5 E6 E7 E8 E9 E45 E11 E12 E16 E5 E5 E17 E14 E13 E15 E14 E14 E14 E47 E18 E4 E14 E3 E17 E19 E19 E20 E43 E14 E47 E24 E52 E25 E23 E22 E53 E14 E28 E31 E31 E32 E33 E34 E35 E26 E29 E30 E36 E27 E55 E38 E37 E39               
E10 E1 E6 E7 E8 E9 E11 E12 E15 E14 E13 E14 E14 E117 E118 E25 E28 E14 E3 E23 E22 E71 E69 E29 E28 E30 E35 E36 E38 E37 E39 E26 E27                                                                                                                                                                         '''

def distance(row1, row2):
    # The function returns the sum of differences between the
    # events in row1 and row2. this is the Manhattan distance.
    # Row1 and row2 are lists with equal length, containing numeric values.
    dist_h = 0
    for i in range (len(row1)):
        dist_h += abs(row1[i]-row2[i])
    return dist_h

def all_pairs(data,N):
    # this calculates the distances between all sentence pairs in the data
    i=0
    j=0
    dist_arr = np.empty((N, N), dtype=float)
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
    dist = np.empty((N, N), dtype=float)
    dist = np.array(all_pairs(data,N))
    print ('-------------------')
    print(dist)   
    print ('-------------------')
    print(np.unravel_index(np.argmin(dist), dist.shape))



def main(events):
    # 1. split the list of events into single events, and get a list of unique events that appear in it
    # a short one-liner to separate the list into a list per shipments (with events lower-cased to make events equal 
    # despite casing) can be done with 
    # docs = [line.lower().split() for line in text.split('\n')]
    docs = [line.lower().split() for line in events.split('\n')]
    #print (docs)
    N = len(docs)
    print (N)
    # create the event vocabulary: the list of events that appear at least once
    vocabulary = list(set(events.lower().split()))
    #print (vocabulary)
    
    # 2. go over each unique events and calculate its term frequency, and its document frequency
    df = {}
    tf = {}  

    for word in vocabulary:
        # tf: number of occurrences of events/word w in document divided by document length
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
    #print (tfidf_array)


main(events)


  


```
## Output of the code in Thonny IDE

Python 3.8.5 (/usr/bin/python3)
>>> %Run shipments-event-comparison.py
19
-------------------
[[       inf 0.11629737 0.30875347 0.09545219 0.28415674 0.04038837
  0.08521573 0.19408161 0.22470077 0.3103889  0.26214163 0.24567429
  0.30808202 0.31547252 0.68751143 0.25510389 0.57998351 0.08439528
  0.22148526]
 [0.11629737        inf 0.34653892 0.13191855 0.32519915 0.08888437
  0.12961487 0.20851158 0.26206049 0.3588235  0.3089825  0.22748898
  0.32460961 0.32949547 0.70622297 0.23378943 0.62087332 0.12553318
  0.27581307]
 [0.30875347 0.34653892        inf 0.26045117 0.09143708 0.28478764
  0.24504644 0.29785372 0.44152124 0.48881249 0.46065168 0.39902122
  0.44133829 0.39937806 0.81435049 0.4474271  0.71708084 0.24488374
  0.42205182]
 [0.09545219 0.13191855 0.26045117        inf 0.25144018 0.07156867
  0.05302773 0.15342663 0.24064229 0.34054585 0.2811132  0.21318601
  0.24501106 0.25009032 0.62845226 0.2478809  0.55780915 0.02400063
  0.24355581]
 [0.28415674 0.32519915 0.09143708 0.25144018        inf 0.26037101
  0.21725516 0.27477102 0.42180587 0.49251143 0.43482287 0.38608404
  0.44627353 0.40740608 0.81736428 0.42660608 0.71196643 0.23462071
  0.39537965]
 [0.04038837 0.08888437 0.28478764 0.07156867 0.26037101        inf
  0.06430708 0.16370308 0.19994883 0.28828223 0.23817131 0.21577263
  0.2825001  0.28614078 0.66831882 0.23136829 0.56066044 0.05874779
  0.19624863]
 [0.08521573 0.12961487 0.24504644 0.05302773 0.21725516 0.06430708
         inf 0.14298641 0.23069458 0.29583687 0.24080176 0.20323658
  0.25675989 0.26378058 0.63781919 0.24843006 0.53648219 0.04039312
  0.20723111]
 [0.19408161 0.20851158 0.29785372 0.15342663 0.27477102 0.16370308
  0.14298641        inf 0.32522409 0.40281923 0.33042813 0.14919463
  0.26587348 0.25862368 0.65818116 0.33401912 0.61500985 0.15974088
  0.30669395]
 [0.22470077 0.26206049 0.44152124 0.24064229 0.42180587 0.19994883
  0.23069458 0.32522409        inf 0.46465481 0.38171463 0.37591477
  0.44355479 0.44340198 0.82278107 0.36670779 0.72027502 0.21855169
  0.37863021]
 [0.3103889  0.3588235  0.48881249 0.34054585 0.49251143 0.28828223
  0.29583687 0.40281923 0.46465481        inf 0.46295543 0.45468627
  0.47278111 0.51883395 0.88202045 0.4013711  0.7481481  0.32255904
  0.35077976]
 [0.26214163 0.3089825  0.46065168 0.2811132  0.43482287 0.23817131
  0.24080176 0.33042813 0.38171463 0.46295543        inf 0.38204354
  0.37028529 0.44564157 0.79749982 0.40962205 0.69577217 0.26604989
  0.37368011]
 [0.24567429 0.22748898 0.39902122 0.21318601 0.38608404 0.21577263
  0.20323658 0.14919463 0.37591477 0.45468627 0.38204354        inf
  0.32761249 0.26893402 0.71344578 0.34053988 0.66921865 0.21881143
  0.35746307]
 [0.30808202 0.32460961 0.44133829 0.24501106 0.44627353 0.2825001
  0.25675989 0.26587348 0.44355479 0.47278111 0.37028529 0.32761249
         inf 0.33998545 0.71617054 0.4469364  0.69173066 0.2560002
  0.42183053]
 [0.31547252 0.32949547 0.39937806 0.25009032 0.40740608 0.28614078
  0.26378058 0.25862368 0.44340198 0.51883395 0.44564157 0.26893402
  0.33998545        inf 0.73467892 0.44714091 0.70622151 0.25825746
  0.41886065]
 [0.68751143 0.70622297 0.81435049 0.62845226 0.81736428 0.66831882
  0.63781919 0.65818116 0.82278107 0.88202045 0.79749982 0.71344578
  0.71617054 0.73467892        inf 0.82456846 0.24268467 0.63580533
  0.77686777]
 [0.25510389 0.23378943 0.4474271  0.2478809  0.42660608 0.23136829
  0.24843006 0.33401912 0.36670779 0.4013711  0.40962205 0.34053988
  0.4469364  0.44714091 0.82456846        inf 0.72412973 0.2398021
  0.34717431]
 [0.57998351 0.62087332 0.71708084 0.55780915 0.71196643 0.56066044
  0.53648219 0.61500985 0.72027502 0.7481481  0.69577217 0.66921865
  0.69173066 0.70622151 0.24268467 0.72412973        inf 0.54338749
  0.66960075]
 [0.08439528 0.12553318 0.24488374 0.02400063 0.23462071 0.05874779
  0.04039312 0.15974088 0.21855169 0.32255904 0.26604989 0.21881143
  0.2560002  0.25825746 0.63580533 0.2398021  0.54338749        inf
  0.22902651]
 [0.22148526 0.27581307 0.42205182 0.24355581 0.39537965 0.19624863
  0.20723111 0.30669395 0.37863021 0.35077976 0.37368011 0.35746307
  0.42183053 0.41886065 0.77686777 0.34717431 0.66960075 0.22902651
         inf]]
-------------------
(3, 17)
>>> 

Of the 19 shipment event sequences the 4th and 18th shipment event profiles are most similar.

## Data sources and AI methods

Data comes from the event log of the transport management system, CargoWise One by WiseTech Global Ltd., of my company SENATOR INTERNATONAL.
Due to confidentiality data is anonymized, real IDs are replaced and event details are generalized.
In the first phase tere is only a reduced subset of event data to work-out the methods of data cleansing and sorting and data analytics.

I have collected manually 19 shipment event log files and stripped them to the necessary data used here for a special trade lane between China (CN) and ermany (DE). 

The data is provided in an ecxel file of following sturcture

| Shipment ID | EVENT Descr. | EventID | Event time stamp 
| ----------- | ------------ | ------- | ----------------
| ID1         | Event 1(ID1) | EvID1   | Time 1 (ID1)
| ...         | ...          | ...     | ...
| ID1         | Event n(ID1) | ...     | Time n (ID1)
| ID2         | Event 1(ID2) | ...     | Time 1 (ID2)
| ...         | ...          | ...     | ...
| ID2         | Event m(ID2) | ...     | Time m (ID2)
| ...         | ...          | ...     | ...
| ...         | ...          | ...     | ...
| IDk         | Event 1(IDk) | ...     | Time 1 (IDk)
| ...         | ...          | ...     | ...
| IDk         | Event p(IDk) | EvIDq   | Time p (IDk)

see attached files:
202104_FCL_CN-DE_Data_anonyn_EventID.ods
Eventlist_as_Text.txt


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
