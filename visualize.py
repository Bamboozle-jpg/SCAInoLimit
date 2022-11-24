import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tsunamList=[*csv.DictReader(open('sources.csv'))]
# Want to look at SOURCE_ID, YEAR, COUNTRY, MAXIMUM_HEIGHT, MAGNITUDE_IIDA and INTENSITY_SOLOVIEV
waveList=[*csv.DictReader(open('waves.csv'))]
# Want to look at SOURCE_ID, WAVE_ID, YEAR, COUNTRY, DISTANCE_FROM_SOURCE, TRAVEL_TIME_HOURS, TRAVEL_TIME_MINUTES
# PERIOD, FIRST_MOTION(What is this?), MAXIMUM_HEIGHT, HORIZONTAL_INUNDATION

fig = plt.figure(tight_layout=True)
ax = fig.add_subplot(2, 2, 1)

print(tsunamList[0])
solov = []
height = []
iida = []
for i in range(0, len(tsunamList)):
    if tsunamList[i]["MAXIMUM_HEIGHT"] and tsunamList[i]["MAGNITUDE_IIDA"] and tsunamList[i]["INTENSITY_SOLOVIEV"]:
        solov.append(float(tsunamList[i]["INTENSITY_SOLOVIEV"]))
        height.append(float(tsunamList[i]["MAXIMUM_HEIGHT"]))
        iida.append(float(tsunamList[i]["MAGNITUDE_IIDA"]))

#Determine what we want to look at for tsunamiList
j = 0
for i in range(0, len(tsunamList)):
    if tsunamList[i]["MAXIMUM_HEIGHT"]:
        j += 1
print("Total with Maximum Height : " + str(j))

j = 0
for i in range(0, len(tsunamList)):
    if tsunamList[i]["MAGNITUDE_IIDA"]:
        j += 1
print("Total with Magnitude IIDA? : " + str(j))

j = 0
for i in range(0, len(tsunamList)):
    if tsunamList[i]["INTENSITY_SOLOVIEV"]:
        j += 1
print("Total with Intensity Soloviev? : " + str(j))

j = 0
for i in range(0, len(tsunamList)):
    if tsunamList[i]["MAXIMUM_HEIGHT"] and tsunamList[i]["MAGNITUDE_IIDA"] and tsunamList[i]["INTENSITY_SOLOVIEV"]:
        print("Source ID : " + tsunamList[i]["SOURCE_ID"])
        if tsunamList[i]['YEAR']:
            print("Year : " + tsunamList[i]['YEAR'])
        if tsunamList[i]['MAXIMUM_HEIGHT']:
            print("Maximum Height : " + tsunamList[i]['MAXIMUM_HEIGHT'])
        if tsunamList[i]['MAGNITUDE_IIDA']:
            print("Magnitude IIDA : " + tsunamList[i]['MAGNITUDE_IIDA'])
        if tsunamList[i]['INTENSITY_SOLOVIEV']:
            print("Intensity Soloviev : " + tsunamList[i]['INTENSITY_SOLOVIEV'])
        print()
        j += 1
print("Total with all : " + str(j) + "\n\n")

# Determine what we want to look at from wave list
j = 0
for i in range(0, len(waveList)):
    if waveList[i]["DISTANCE_FROM_SOURCE"]:
        j += 1
print("Total with Dist from source : " + str(j))

# Want to take TRAVEL_TIME_HOURS and TRAVEL_TIME_MINUTES for this one
j = 0
for i in range(0, len(waveList)):
    if waveList[i]["TRAVEL_TIME_HOURS"]:
        j += 1
print("Total with Travel Time : " + str(j))

j = 0
for i in range(0, len(waveList)):
    if waveList[i]["PERIOD"]:
        j += 1
print("Total with period : " + str(j))

j = 0
for i in range(0, len(waveList)):
    if waveList[i]["FIRST_MOTION"]:
        j += 1
print("Total with First Motion : " + str(j))

# for i in range(0, len(waveList)):
#     if(waveList[i]["FIRST_MOTION"]):
#         print(waveList[i]["FIRST_MOTION"])

# Gives F or R? and Only 1.5k so IDK

j = 0
for i in range(0, len(waveList)):
    if waveList[i]["MAXIMUM_HEIGHT"]:
        j += 1
print("Total with Max Height : " + str(j))

j = 0
for i in range(0, len(waveList)):
    if waveList[i]["HORIZONTAL_INUNDATION"]:
        j += 1
print("Total with horiz Inundation : " + str(j))

j = 0
for i in range(0, len(waveList)):
    if waveList[i]["SOURCE_ID"]:
        j += 1
print("Total : " + str(j))

solov2 = []
wHeight = []
wInundate = []

j = 0
for i in range(0, len(waveList)):
    if waveList[i]["MAXIMUM_HEIGHT"]:
        if waveList[i]["HORIZONTAL_INUNDATION"]:
            identity = waveList[i]["SOURCE_ID"]
            for k in range(0, len(tsunamList)):
                if tsunamList[k]["SOURCE_ID"] == identity:
                    if tsunamList[k]["INTENSITY_SOLOVIEV"]:
                        if float(waveList[i]["MAXIMUM_HEIGHT"]) < 35 and float(waveList[i]["HORIZONTAL_INUNDATION"]) < 4500:
                            j += 1
                            solov2.append(float(tsunamList[k]["INTENSITY_SOLOVIEV"]))
                            wHeight.append(float(waveList[i]["MAXIMUM_HEIGHT"]))
                            wInundate.append(float(waveList[i]["HORIZONTAL_INUNDATION"]))
print("Total with Max Height and Horiz Inundate and solov Tsunami : " + str(j))

'''
{'SOURCE_ID': '1', 'WAVE_ID': '11014', 'YEAR': '-2000', 'COUNTRY': 'SYRIA', 'DISTANCE_FROM_SOURCE': '12', 
'TRAVEL_TIME_HOURS': '', 'PERIOD': '', 'FIRST_MOTION': '', 'MAXIMUM_HEIGHT': '', 'HORIZONTAL_INUNDATION': ''
'''

ax.scatter(solov2, wInundate, s=15,color='orange')
ax.set_title("Scatter Plot")
ax.set_xlabel('Soloviev Intensity')
ax.set_ylabel('Horizontal Inundation')

ax = fig.add_subplot(2, 2, 2)
ax.scatter(solov2, wHeight, s=15,color='green')
ax.set_title("Scatter Plot")
ax.set_xlabel('Soloviev Intensity')
ax.set_ylabel('Indiv wave height')
# ax.set_xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
# y_labels = { -1 : ".1", 0 : "1", 1 : "10", 2 : "100" }
# ax.set_yticks(list(y_labels.keys()), y_labels.values())

colors = []
for i in range(0, len(solov2)):
    if solov2[i] <= 1:
        colors.append('red')
    elif solov2[i] <= 2:
        colors.append('green')
    elif solov2[i] <= 3:
        colors.append('blue')
    elif solov2[i] <= 4:
        colors.append('orange')
    elif solov2[i] <= 5:
        colors.append('yellow')

print(len(solov2))
print(len(colors))
    
ax = fig.add_subplot(2, 2, 4)
ax.scatter(wInundate, wHeight, s=2,color=colors)
ax.set_title("Scatter Plot")
ax.set_xlabel('Horizontal Inundation')
ax.set_ylabel('Indiv wave height')

data={'x':wHeight,
      'y':wInundate,
      'z':solov2
     }

df = pd.DataFrame(data)
data_np=df.to_numpy()

x = data_np[:,0]
y = data_np[:,1]
z = data_np[:,2]

z2=np.ones(shape=x.shape)*min(z)

ax = fig.add_subplot(2, 2, 3, projection='3d')

#scatter
sc = ax.scatter(x, y, z, s=0, marker='o',color='red', alpha=1)

# lines
for i,j,k,h in zip(wHeight,wInundate,solov2,z2):
    if k <= 1:
        ax.plot([i,i],[j,j],[k,h], color='red')
    elif k <= 2:
        ax.plot([i,i],[j,j],[k,h], color='green')
    elif k <= 3:
        ax.plot([i,i],[j,j],[k,h], color='blue')
    elif k <= 4:
        ax.plot([i,i],[j,j],[k,h], color='orange')
    elif k <= 5:
        ax.plot([i,i],[j,j],[k,h], color='yellow')

ax.set_xlabel('Wave Height Max')
ax.set_ylabel('Horizontal Inundation')
ax.set_zlabel('Soloviev Identity')

csv_columns = ['waveHeight', 'horizontalInundation', 'solovievIdentity']
retList = []
for i in range(len(wHeight)):
    retDict = {}
    retDict['waveHeight'] = wHeight[i]
    retDict['horizontalInundation'] = wInundate[i]
    retDict['solovievIdentity'] = solov2[i]
    
    retList.append(retDict)

print(retList)

# POSTS STUFF TO FINAL DATA, DON'T USE YET CUZ IT WILL FUCK UP DATA STUFF
# with open('finalData.csv', 'a') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames = csv_columns)
#     writer.writeheader()
#     for row in retList:
#         writer.writerow(row)

plt.show()