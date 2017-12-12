#!/usr/bin/env python
import os
import subprocess as sp
from datetime import datetime

problemSize = 1000000
output = []

# FOR EACH TARGET OCCPUANCY 5-100, 5% INCREMENTS
for i in range(1,11): # 10, 20, 30, ..., 90, 100%
	targetOccupancy = (i * 10)
	# FOR EACH OCCUPANCY METHOD
	for i2 in range(0,2): # 0, 1
		# FOR EACH WORK FUNCTION
		for i3 in range(0,3): # 0, 1, 2
			times = []
			# RUN TEST 9X
			for i4 in range(0,9):
				start = datetime.now()
				sp.call(['./t.exe', str(i2), str(i3), str(targetOccupancy), str(problemSize)])
				end = datetime.now()
				times.append(((end - start).microseconds))
			# SAVE MEDIAN AS: "METHOD,TARGETOCCUPANCY,TIME"
			times.sort()
			output.append("%s,%s,%s,%s" %(str(i2), str(i3), str(targetOccupancy), str(times[4])))

print("Occupancy Method, Function Used, Target Occupancy, Median Time")
for x in output:
	print("%s" %(x))