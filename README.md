# Desciption: An interactive model and map that tracks any earthquakes that may occur in a specific region using real time data.

## Key Feature:
The circle scattered across the map represent earthquake activity that has been detected in that specific area. Each of the numbers in the circle represent the possible number of earthquakes that have been detected in that area. Additionally, large numbers repressnt a higher seismic activity, with the various colors representing the intensity of each detected earthquake. Green represents fewer or less intensive earthquakes, whereas yellow, red, and orange represent stronger and increasing seismic activity.

## Our algorithm:
All the data present in the program is accurate and is garnered from the USGS Earthquake API. The preprocessing, which basically sets the presets for each earthquake detection, showcases time, longitude, depth, magnitude, and location. Additiaionyll, we used a histogram to visually showcase the magnitudes of earthquakes with their respective mean and median. Our interactive map uses the folium model in order to assign each earthquake cluster to a specific color. 
The detector uses Machine Learning, in terms of RandomForest from the sklearn module, which ultimately measures the F1 score, confusion matrix, and reports. RandomForest handles binary data in order to allow geospatial and temporal data to co-exist. Furthermore, in order to complete the full model of the detector, we had to train and evaluate the model, as it was split in x and y targets, where earthquakes are considered as significant if their magnitude is greater than or equal to 5.

![image](https://github.com/user-attachments/assets/7dbaf844-1a82-4622-9e0d-b14bfeb85bc5)


