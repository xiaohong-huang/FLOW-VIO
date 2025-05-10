import subprocess
import os
import multiprocessing
import psutil
import time
import random
import threading



def run_func():
    for i in range(len(filenames)):
        try:

            f_idx=i
            print("rosrun", "flow_vio","4seasons_convert2bag_node",config_name,
                                    filenames[f_idx]+".bag",filenames[f_idx]+"/imu.txt",filenames[f_idx]+"/undistorted_images")

            result = subprocess.run(["rosrun", "flow_vio","4seasons_convert2bag_node",config_name,
                                filenames[f_idx]+".bag",filenames[f_idx]+"/imu.txt",filenames[f_idx]+"/undistorted_images"], stdout=subprocess.PIPE)

            a=result.stdout.decode().split("\n")
            print(a)


        except Exception as ret:
            print(ret)
            


folder="/media/huang/新加卷1/dataset-public/4seasons/"

filenames=["business_2020-10-08_09-30-57",
"business_2021-01-07_13-12-23",
"business_2021-02-25_14-16-43",
"city_2020-12-22_11-33-15",
"city_2021-01-07_14-36-17",
"city_2021-02-25_11-09-49",
"country_2020-04-07_11-33-45",
"country_2020-06-12_11-26-43",
"country_2020-10-08_09-57-28",
"country_2021-01-07_13-30-07",
"neighbor_2020-03-26_13-32-55",
"neighbor_2020-10-07_14-47-51",
"neighbor_2020-10-07_14-53-52",
"neighbor_2020-12-22_11-54-24",
"neighbor_2021-02-25_13-25-15",
"neighbor_2021-05-10_18-02-12",
"neighbor_2021-05-10_18-32-32",
"office_2020-03-24_17-36-22",
"office_2020-03-24_17-45-31",
"office_2020-04-07_10-20-32",
"office_2020-06-12_10-10-57",
"office_2021-01-07_12-04-03",
"office_2021-02-25_13-51-57",
"oldtown_2020-10-08_11-53-41",
"oldtown_2021-01-07_10-49-45",
"oldtown_2021-02-25_12-34-08",
"oldtown_2021-05-10_21-32-00",
"parking_2020-12-22_12-04-35",
"parking_2021-02-25_13-39-06",
"parking_2021-05-10_19-15-19",
]
filenames=[folder+f for f in filenames]
config_name="src/FLOW-VIO/yaml/4seasons_config.yaml"
run_func()
