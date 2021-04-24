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


Images will make your README look nice!
Once you upload an image to your repository, you can link link to it like this (replace the URL with file path, if you've uploaded an image to Github.)
![Container vessel](https://pixabay.com/de/photos/hamburg-hafen-schiff-container-212405/)

If you need to resize images, you have to use an HTML tag, like this:
<img src="https://pixabay.com/de/photos/hamburg-hafen-schiff-container-212405/" width="300">

This is how you create code examples:
```
def main():
   countries = ['Denmark', 'Finland', 'Iceland', 'Norway', 'Sweden']
   pop = [5615000, 5439000, 324000, 5080000, 9609000]   # not actually needed in this exercise...
   fishers = [1891, 2652, 3800, 11611, 1757]

   totPop = sum(pop)
   totFish = sum(fishers)

   # write your solution here

   for i in range(len(countries)):
      print("%s %.2f%%" % (countries[i], 100.0))    # current just prints 100%

main()
```


## Data sources and AI methods
Where does your data come from? Do you collect it yourself or do you use data collected by someone else?
If you need to use links, here's an example:
[Twitter API](https://developer.twitter.com/en/docs)

| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |

## Challenges

What does your project _not_ solve? Which limitations and ethical considerations should be taken into account when deploying a solution like this?

## What next?

How could your project grow and become something even more? What kind of skills, what kind of assistance would you  need to move on? 


## Acknowledgments

* list here the sources of inspiration 
* do not use code, images, data etc. from others without permission
* when you have permission to use other people's materials, always mention the original creator and the open source / Creative Commons licence they've used
  <br>For example: [Sleeping Cat on Her Back by Umberto Salvagnin](https://commons.wikimedia.org/wiki/File:Sleeping_cat_on_her_back.jpg#filelinks) / [CC BY 2.0](https://creativecommons.org/licenses/by/2.0)
* etc
