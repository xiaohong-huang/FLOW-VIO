import subprocess


process_num=0
def process_item(i):
    global process_num

    f_idx=i%f_size+1
    print("rosrun", "flow_vio","flow_vio_node",config_name,
                            filenames[f_idx-1],prefix,str(f_idx))
    result = subprocess.run(["rosrun", 
                            "flow_vio",
                            "flow_vio_node",
                            config_name,
                            filenames[f_idx-1],
                            prefix+"_"+str(f_idx)+".txt"], stdout=subprocess.PIPE)
    a=result.stdout.decode().split("\n")

    if(a[-1]=="finish"or a[-2]=="finish" or a[-3]=="finish"):
        print(filenames[f_idx-1],prefix,str(f_idx),"finish!!!")
    else:
        print("fail",filenames[f_idx-1],prefix,str(f_idx))
        return






def run_func():
    global process_num
    for j in range(len(filenames)):
        process_item(j)





filenames=["/media/huang/新加卷1/dataset-public/euroc/MH_01_easy.bag",
"/media/huang/新加卷1/dataset-public/euroc/MH_02_easy.bag",
"/media/huang/新加卷1/dataset-public/euroc/MH_03_medium.bag",
"/media/huang/新加卷1/dataset-public/euroc/MH_04_difficult.bag",
"/media/huang/新加卷1/dataset-public/euroc/MH_05_difficult.bag",
"/media/huang/新加卷1/dataset-public/euroc/V1_01_easy.bag",
"/media/huang/新加卷1/dataset-public/euroc/V1_02_medium.bag",
"/media/huang/新加卷1/dataset-public/euroc/V1_03_difficult.bag",
"/media/huang/新加卷1/dataset-public/euroc/V2_01_easy.bag",
"/media/huang/新加卷1/dataset-public/euroc/V2_02_medium.bag",
"/media/huang/新加卷1/dataset-public/euroc/V2_03_difficult.bag"]

f_size=len(filenames)
prefix="EUROC"
config_name="src/FLOW-VIO/yaml/euroc_config.yaml"
run_func()


filenames=[
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors1_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors2_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors3_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors4_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors5_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors6_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors7_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors8_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-corridor1_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-corridor2_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-corridor3_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-corridor4_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-corridor5_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-magistrale1_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-magistrale2_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-magistrale3_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-magistrale4_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-magistrale5_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-magistrale6_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-room1_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-room2_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-room3_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-room4_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-room5_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-room6_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-slides1_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-slides2_512_16.bag",
"/media/huang/新加卷1/dataset-public/tum/dataset-slides3_512_16.bag"
]
f_size=len(filenames)
prefix="TUM"
config_name="src/FLOW-VIO/yaml/tum_config.yaml"
run_func()





filenames=[
"/media/huang/新加卷1/dataset-public/my/R1M1.bag",
"/media/huang/新加卷1/dataset-public/my/R2M1.bag",
"/media/huang/新加卷1/dataset-public/my/R1M2.bag",
"/media/huang/新加卷1/dataset-public/my/R2M2.bag",
]
f_size=len(filenames)
prefix="MY"
config_name="src/FLOW-VIO/yaml/my_config.yaml"
run_func()



filenames=[
"/media/huang/新加卷/4seasons/office_2021-01-07_12-04-03.bag",#1
"/media/huang/新加卷/4seasons/office_2021-02-25_13-51-57.bag",
"/media/huang/新加卷/4seasons/office_2020-03-24_17-36-22.bag",
"/media/huang/新加卷/4seasons/office_2020-03-24_17-45-31.bag",
"/media/huang/新加卷/4seasons/office_2020-04-07_10-20-32.bag",
"/media/huang/新加卷/4seasons/office_2020-06-12_10-10-57.bag",
"/media/huang/新加卷/4seasons/neighbor_2020-10-07_14-47-51.bag",#7
"/media/huang/新加卷/4seasons/neighbor_2020-10-07_14-53-52.bag",
"/media/huang/新加卷/4seasons/neighbor_2020-12-22_11-54-24.bag",
"/media/huang/新加卷/4seasons/neighbor_2021-02-25_13-25-15.bag",
"/media/huang/新加卷/4seasons/neighbor_2020-03-26_13-32-55.bag",
"/media/huang/新加卷/4seasons/neighbor_2021-05-10_18-02-12.bag",
"/media/huang/新加卷/4seasons/neighbor_2021-05-10_18-32-32.bag",
"/media/huang/新加卷/4seasons/business_2021-01-07_13-12-23.bag",#14
"/media/huang/新加卷/4seasons/business_2021-02-25_14-16-43.bag",
"/media/huang/新加卷/4seasons/business_2020-10-08_09-30-57.bag",
"/media/huang/新加卷/4seasons/country_2020-10-08_09-57-28.bag",#17
"/media/huang/新加卷/4seasons/country_2021-01-07_13-30-07.bag",
"/media/huang/新加卷/4seasons/country_2020-04-07_11-33-45.bag",
"/media/huang/新加卷/4seasons/country_2020-06-12_11-26-43.bag",
"/media/huang/新加卷/4seasons/city_2020-12-22_11-33-15.bag",#21
"/media/huang/新加卷/4seasons/city_2021-01-07_14-36-17.bag",
"/media/huang/新加卷/4seasons/city_2021-02-25_11-09-49.bag",
"/media/huang/新加卷/4seasons/oldtown_2020-10-08_11-53-41.bag",#24
"/media/huang/新加卷/4seasons/oldtown_2021-01-07_10-49-45.bag",
"/media/huang/新加卷/4seasons/oldtown_2021-02-25_12-34-08.bag",
"/media/huang/新加卷/4seasons/oldtown_2021-05-10_21-32-00.bag",
"/media/huang/新加卷/4seasons/parking_2020-12-22_12-04-35.bag",#28
"/media/huang/新加卷/4seasons/parking_2021-02-25_13-39-06.bag",
"/media/huang/新加卷/4seasons/parking_2021-05-10_19-15-19.bag",
]
f_size=len(filenames)
prefix="4SEASONS"
config_name="src//FLOW-VIO/yaml/4seasons_config.yaml"
run_func()