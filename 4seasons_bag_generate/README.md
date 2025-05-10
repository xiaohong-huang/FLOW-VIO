# Gererating ROS bag of the 4Seasons dataset



###  Step 1
In 4seasons_bag_generate.py, you may need to change the "folder" to be the folder names of the 4seasons dataset.
###  Step 2
Run with all datasets:
```
cd ~/catkin_ws/
cp ./src/FLOW-VIO/4seasons_bag_generate/4seasons_bag_generate.py ./4seasons_bag_generate.py
source devel/setup.bash
python3 4seasons_bag_generate.py
```

