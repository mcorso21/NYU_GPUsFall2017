#!/usr/bin/env python
import os
import subprocess as sp
from datetime import datetime

problemSize = 100000
output = []

# FOR EACH TARGET OCCPUANCY 5-100, 5% INCREMENTS
for i in range(1,21):
	targetOccupancy = (i * 5)
	# FOR EACH OCCUPANCY METHOD
	for i2 in range(0,3):
		times = []
		# RUN TEST 9X
		for i3 in range(0,9):
			start = datetime.now()
			sp.call(['./t.exe', str(i2), str(targetOccupancy), str(problemSize)])
			end = datetime.now()
			times.append(((end - start).microseconds))
		# SAVE MEDIAN AS: "METHOD,TARGETOCCUPANCY,TIME"
		times.sort()
		output.append("%s,%s,%s" %(str(i2), str(targetOccupancy), str(times[4])))

print("Occupancy Method, Target Occupancy, Median Time")
for x in output:
	print("%s" %(x))