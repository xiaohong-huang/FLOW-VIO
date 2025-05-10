import subprocess


def evaluate_fun():
    for fi in (range(len(truth))):
        try:
            if use_small_time_diff:
                result = subprocess.run(["evo_ape", "tum",
                                        truth[fi],
                                        prefix+"_"+str(fi+1)+".txt", "-va","--t_max_diff","0.1"], stdout=subprocess.PIPE)
            else:
                result = subprocess.run(["evo_ape", "tum",
                                        truth[fi],
                                        prefix+"_"+str(fi+1)+".txt", "-va"], stdout=subprocess.PIPE)
            a=result.stdout.decode()
            length=a.split("\n")[2].split(" ")[1]
            rmse=a.split("\n")[-5].split("\t")[1]
            print(prefix+"_"+str(fi+1)+":",rmse)
        except:
            print(a)


truth=[
"/media/huang/新加卷1/dataset-public/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.tum",
"/media/huang/新加卷1/dataset-public/euroc/MH_02_easy/mav0/state_groundtruth_estimate0/data.tum",
"/media/huang/新加卷1/dataset-public/euroc/MH_03_medium/mav0/state_groundtruth_estimate0/data.tum",
"/media/huang/新加卷1/dataset-public/euroc/MH_04_difficult/mav0/state_groundtruth_estimate0/data.tum",
"/media/huang/新加卷1/dataset-public/euroc/MH_05_difficult/mav0/state_groundtruth_estimate0/data.tum",
"/media/huang/新加卷1/dataset-public/euroc/V1_01_easy/mav0/state_groundtruth_estimate0/data.tum",
"/media/huang/新加卷1/dataset-public/euroc/V1_02_medium/mav0/state_groundtruth_estimate0/data.tum",
"/media/huang/新加卷1/dataset-public/euroc/V1_03_difficult/mav0/state_groundtruth_estimate0/data.tum",
"/media/huang/新加卷1/dataset-public/euroc/V2_01_easy/mav0/state_groundtruth_estimate0/data.tum",
"/media/huang/新加卷1/dataset-public/euroc/V2_02_medium/mav0/state_groundtruth_estimate0/data.tum",
"/media/huang/新加卷1/dataset-public/euroc/V2_03_difficult/mav0/state_groundtruth_estimate0/data.tum"]
prefix="EUROC"
round_size=3
outdoors=[]
indoors=[i for i in range(len(truth)) if i not in outdoors]
use_small_time_diff=0
evaluate_fun()


truth=[
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors1_512_16/dataset-outdoors1_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors2_512_16/dataset-outdoors2_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors3_512_16/dataset-outdoors3_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors4_512_16/dataset-outdoors4_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors5_512_16/dataset-outdoors5_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors6_512_16/dataset-outdoors6_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors7_512_16/dataset-outdoors7_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-outdoors8_512_16/dataset-outdoors8_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-corridor1_512_16/dataset-corridor1_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-corridor2_512_16/dataset-corridor2_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-corridor3_512_16/dataset-corridor3_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-corridor4_512_16/dataset-corridor4_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-corridor5_512_16/dataset-corridor5_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-magistrale1_512_16/dataset-magistrale1_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-magistrale2_512_16/dataset-magistrale2_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-magistrale3_512_16/dataset-magistrale3_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-magistrale4_512_16/dataset-magistrale4_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-magistrale5_512_16/dataset-magistrale5_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-magistrale6_512_16/dataset-magistrale6_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-room1_512_16/dataset-room1_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-room2_512_16/dataset-room2_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-room3_512_16/dataset-room3_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-room4_512_16/dataset-room4_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-room5_512_16/dataset-room5_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-room6_512_16/dataset-room6_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-slides1_512_16/dataset-slides1_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-slides2_512_16/dataset-slides2_512_16/dso/gt_imu.tum",
"/media/huang/新加卷1/dataset-public/tum/dataset-slides3_512_16/dataset-slides3_512_16/dso/gt_imu.tum"]
prefix="TUM"
indoors=[i for i in range(len(truth)) if i not in outdoors]
use_small_time_diff=0
evaluate_fun()



truth=[
"/media/huang/新加卷1/dataset-public/my/R1M1_2.csv",
"/media/huang/新加卷1/dataset-public/my/R2M1_2.csv",
"/media/huang/新加卷1/dataset-public/my/R1M2_2.csv",
"/media/huang/新加卷1/dataset-public/my/R2M2_2.csv",
]
prefix="MY"
use_small_time_diff=1
evaluate_fun()


truth=[
"/media/huang/新加卷/4seasons/office_2021-01-07_12-04-03/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/office_2021-02-25_13-51-57/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/office_2020-03-24_17-36-22/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/office_2020-03-24_17-45-31/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/office_2020-04-07_10-20-32/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/office_2020-06-12_10-10-57/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/neighbor_2020-10-07_14-47-51/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/neighbor_2020-10-07_14-53-52/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/neighbor_2020-12-22_11-54-24/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/neighbor_2021-02-25_13-25-15/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/neighbor_2020-03-26_13-32-55/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/neighbor_2021-05-10_18-02-12/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/neighbor_2021-05-10_18-32-32/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/business_2021-01-07_13-12-23/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/business_2021-02-25_14-16-43/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/business_2020-10-08_09-30-57/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/country_2020-10-08_09-57-28/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/country_2021-01-07_13-30-07/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/country_2020-04-07_11-33-45/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/country_2020-06-12_11-26-43/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/city_2020-12-22_11-33-15/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/city_2021-01-07_14-36-17/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/city_2021-02-25_11-09-49/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/oldtown_2020-10-08_11-53-41/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/oldtown_2021-01-07_10-49-45/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/oldtown_2021-02-25_12-34-08/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/oldtown_2021-05-10_21-32-00/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/parking_2020-12-22_12-04-35/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/parking_2021-02-25_13-39-06/GNSSPoses_IMU.txt_tum",
"/media/huang/新加卷/4seasons/parking_2021-05-10_19-15-19/GNSSPoses_IMU.txt_tum",
]
prefix="4SEASONS"
use_small_time_diff=0
evaluate_fun()



