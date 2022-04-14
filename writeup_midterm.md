# Sensor Fusion and Tracking Mid-Term

This is the Mid-Term Project for the second course in the [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213) : Sensor Fusion and Tracking.

In this project, real-world data from [Waymo Open Dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files) and 3D Point Cloud are used for LiDAR based Object Detection.

## Project Sections

1. Compute Lidar Point-Cloud from Range Image
2. Create Birds-Eye View from Lidar PCL
3. Model-based Object Detection in BEV Image
4. Performance Evaluation for Object Detection

To run this project:
```
python loop_over_dataset.py
```

### Section 1: Compute Lidar Point-Cloud from Range Image
This section contains 2 steps:

1. Visualize range image channels (ID_S1_EX1)
2. Visualize lidar point-cloud (ID_S1_EX2)


#### Step 1: Visualize range image channels (ID_S1_EX1)
In this step, two of the data channels within the range image `range` and `intensity` are extracted and converted the floating-point data to an 8-bit integer value range.

The changes are made in `loop_over_dataset.py`:

```
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
show_only_frames = [0, 1]
sequence = "1"
exec_data = []
exec_detection = []
exec_tracking = []
exec_visualization = ['show_range_image']
```

The function `show_range image` in `objdet_pcl.py` is implemented:

```
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]
    ri = dataset_pb2.MatrixFloat()
    ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
    ri = np.array(ri.data).reshape(ri.shape.dims)

    # step 2 : extract the range and the intensity channel from the range image
    ri_range = ri[:,:,0]
    ri_intensity = ri[:,:,1]

    # step 3 : set values <0 to zero
    ri[ri<0] = 0.0

    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))
    img_range = ri_range.astype(np.uint8)

    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    percentile_1, percetile_99 = np.percentile(ri_intensity, 1), np.percentile(ri_intensity, 99)
    ri_intensity = 255 * np.clip(ri_intensity, percentile_1, percetile_99)/percetile_99
    img_intensity = ri_intensity.astype(np.uint8)

    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    img_range_intensity = np.vstack((ri_range, ri_intensity)).astype(np.uint8)
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity
```

The result is:

<img src="/img/S1_EX1.png"/>

#### Step 2: Visualize lidar point-cloud (ID_S1_EX2)
In this step, the Open3D library is used to display the lidar point-cloud in a 3D viewer.

The changes are made in `loop_over_dataset.py`:

```
data_filename = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord'
show_only_frames = [0, 200]
sequence = "3"
exec_data = []
exec_detection = []
exec_tracking = []
exec_visualization = ['show_pcl']
```

The function `show_pcl` in `objdet_pcl.py` is implemented:

```
def show_pcl(pcl):

    ####### ID_S1_EX2 START #######     
    #######
    print("student task ID_S1_EX2")

    # step 1 : initialize open3d with key callback and create window
    window = open3d.visualization.VisualizerWithKeyCallback()
    window.create_window(window_name='student task ID_S1_EX2')
    
    def key(window):
        window.close()
        return False

    # step 2 : create instance of open3d point-cloud class
    pcd = open3d.geometry.PointCloud()

    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    pcd.points = open3d.utility.Vector3dVector(pcl[:,:3])

    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    window.add_geometry(pcd)
    
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    window.register_key_callback(262, key)
    window.run()
    
    #######
    ####### ID_S1_EX2 END ####### 
```

The result is:

<img src="/img/S1_EX2_1.png"/>
<img src="/img/S1_EX2_2.png"/>
<img src="/img/S1_EX2_3.png"/>
<img src="/img/S1_EX2_4.png"/>


### Section 2: Create Birds-Eye View from Lidar PCL
This section contains 3 steps:

1. Convert sensor coordinates to BEV-map coordinates (ID_S2_EX1)
2. Compute intensity layer of the BEV map (ID_S2_EX2)
3. Compute height layer of the BEV map (ID_S2_EX3)

#### Step 1: Convert sensor coordinates to BEV-map coordinates (ID_S2_EX1)
The goal of this step is creating a birds-eye view (BEV) perspective of the lidar point-cloud.

The changes are made in `loop_over_dataset.py`:
```
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
show_only_frames = [0, 1]
sequence = "1"
exec_data = ['pcl_from_rangeimage']
exec_detection = ['bev_from_pcl']
exec_tracking = []
exec_visualization = []
```

The function `bev_from_pcl` in `objdet_pcl.py` is implemented:

```
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / discret))

    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / discret) + (configs.bev_width + 1) / 2)
    lidar_pcl_cpy[:, 1] = np.abs(lidar_pcl_cpy[:,1])

    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    show_pcl(lidar_pcl_cpy)
    #######
    ####### ID_S2_EX1 END #######   
```

The result is:

<img src="/img/S2_EX1.png"/>

#### Step 2: Compute intensity layer of the BEV map (ID_S2_EX2)
The goal of this step is to fill the `intensity` channel of the BEV map with data from the point-cloud.

The changes are made in `loop_over_dataset.py`:

```
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
show_only_frames = [0, 1]
sequence = "1" 
exec_data = ['pcl_from_rangeimage']
exec_detection = ['bev_from_pcl']
exec_tracking = []
exec_visualization = []
```

The function `bev_from_pcl` in `objdet_pcl.py` is updated:

```
####### ID_S2_EX2 START #######     
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs.bev_height, configs.bev_width))

    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    lidar_pcl_cpy[lidar_pcl_cpy[:, 3]>1.0, 3] = 1.0
    idx_intensity = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[: ,1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_cpy = lidar_pcl_cpy[idx_intensity]

    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _, idxs, count = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    lidar_pcl_top = lidar_pcl_cpy[idxs]

    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    percentile_1 = np.percentile(lidar_pcl_top[:, 3], 1)
    percetile_99 = np.percentile(lidar_pcl_top[:, 3], 99)
    intensity_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = (np.clip(lidar_pcl_top[:, 3], percentile_1, percetile_99) - percentile_1) / (percetile_99 - percentile_1)
    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    img_intensity = intensity_map * 256
    img_intensity = img_intensity.astype(np.uint8)
    cv2.imshow('Intensity', img_intensity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #######
    ####### ID_S2_EX2 END #######
```

The result is:

<img src="/img/S2_EX2.png"/>

#### Step 3: Compute height layer of the BEV map (ID_S2_EX3)

The goal of this step is to fill the `height` channel of the BEV map with data from the point-cloud.

The changes are made in `loop_over_dataset.py`:

```
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
show_only_frames = [0, 1]
sequence = "1" 
exec_data = ['pcl_from_rangeimage']
exec_detection = ['bev_from_pcl']
exec_tracking = []
exec_visualization = []
```

The function `bev_from_pcl` in `objdet_pcl.py` is updated:

```
####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros((configs.bev_height, configs.bev_width))

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
    height_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 2] / float(configs.lim_z[1] - configs.lim_z[0])

    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    img_height = height_map * 256
    img_height = img_height.astype(np.uint8)
    cv2.imshow('Height Map', height_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #######
    ####### ID_S2_EX3 END #######    
```

The result is:

<img src="/img/S2_EX3.png"/>

### Section 3: Model-based Object Detection in BEV Image
This section contains 2 steps:

1. Add a second model from a GitHub repo (ID_S3_EX1)
2. Extract 3D bounding boxes from model response (ID_S3_EX2)

#### Step 1: Add a second model from a GitHub repo (ID_S3_EX1)

The goal of step task is to illustrate how a new model can be integrated into an existing framework.

The changes are made in `loop_over_dataset.py`:

```
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
show_only_frames = [50, 51]
model = "fpn-resnet"
model_seq = "resnet"
results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model_seq)
sequence = "1"
exec_data = ['pcl_from_rangeimage', 'load_image']
exec_detection = ['bev_from_pcl', 'detect_objects']
exec_tracking = []
exec_visualization = ['show_objects_in_bev_labels_in_camera']
configs_det = det.load_configs(model_name="fpn_resnet")
```

The changes are made in function `load_configs_model` in `objdet_detect.py`:

```
elif model_name == 'fpn_resnet':
        ####### ID_S3_EX1-3 START #######     
        #######
        print("student task ID_S3_EX1-3")
        configs.arch = 'fpn_resnet'
        configs.saved_fn = 'fpn_resnet'
        configs.pretrained_path = 'tools/objdet_models/resnet/pretrained/fpn_resnet_18_epoch_300.pth'
        configs.k = 50
        configs.conf_thresh = 0.5
        configs.no_cuda = False
        configs.gpu_idx = 0
        configs.batch_size = 1
        configs.num_samples = None
        configs.num_workers = 1
        configs.peak_thresh = 0.2
        configs.save_test_output = False
        configs.output_format = 'image'
    
        configs.output_width = 608
        configs.pin_memory = True
        configs.distributed = False
        configs.input_size = (608,608)
        configs.hm_size = (152,152)
        configs.down_ratio = 4
        configs.max_objects = 50
        configs.imagenet_pretrained = False
        configs.head_conv = 64
        configs.num_classes = 3
        configs.num_center_offset = 2
        configs.num_z = 1
        configs.num_dim = 3
        configs.num_direction = 2
        
        configs.heads = {
            'hm_cen': configs.num_classes, 
            'cen_offset': configs.num_center_offset,
            'direction': configs.num_direction,
            'z_coor': configs.num_z,
            'dim': configs.num_dim
        }

        configs.num_input_features = 4

        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'resnet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')

        #######
        ####### ID_S3_EX1-3 END #######
```

The changes are made in function `create_model` in `objdet_detect.py`:

```
elif 'fpn_resnet' in configs.arch:
        print('using ResNet architecture with feature pyramid')
        
        ####### ID_S3_EX1-4 START #######     
        #######
        print("student task ID_S3_EX1-4")
        model = fpn_resnet.get_pose_net(num_layers = 18, heads = configs.heads, head_conv = configs.head_conv, imagenet_pretrained = configs.imagenet_pretrained)
        #######
        ####### ID_S3_EX1-4 END ####### 
```

The changes are made in function `detect_objects` in `objdet_detect.py`:

```
elif 'fpn_resnet' in configs.arch:
            # decode output and perform post-processing
            
            ####### ID_S3_EX1-5 START #######     
            #######
            print("student task ID_S3_EX1-5")
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])

            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'], outputs['dim'], K=40)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs)
            detections = detections[0][1]
            #######
            ####### ID_S3_EX1-5 END #######   
```

#### Step 2: Extract 3D bounding boxes from model response (ID_S3_EX2)
As the model input is a three-channel BEV map, the detected objects will be returned with coordinates and properties in the BEV coordinate space. Thus, before the detections can move along in the processing pipeline, they need to be converted into metric coordinates in vehicle space.

The changes are made in `loop_over_dataset.py`:

```
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
show_only_frames = [50, 51]
model = "fpn-resnet"
model_seq = "resnet"
results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model_seq)
sequence = "1"
exec_data = ['pcl_from_rangeimage', 'load_image']
exec_detection = ['bev_from_pcl', 'detect_objects']
exec_tracking = []
exec_visualization = ['show_objects_in_bev_labels_in_camera']
configs_det = det.load_configs(model_name="fpn_resnet")
```

The function `detect_objects` is updated:

```
####### ID_S3_EX2 START #######     
    #######
    # Extract 3d bounding boxes from model response
    print("student task ID_S3_EX2")
    objects = [] 

    ## step 1 : check whether there are any detections
    if len(detections) == 0:
        return objects
    
    ## step 2 : loop over all detections
    for obj in detections:
        id, bev_x, bev_y, z, h, bev_w, bev_l, yaw = obj
        
        ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
        x = bev_y / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0]) + configs.lim_x[0]
        y = bev_x / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0]) + configs.lim_y[0]
        z = z + configs.lim_z[0]
        w = bev_w / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0])
        l = bev_l / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
        
        ## step 4 : append the current object to the 'objects' array
        objects.append([1, x, y, z, h, w, l, -yaw])
    #######
    ####### ID_S3_EX2 START #######  
```

The result is:

<img src="/img/S3_EX2.png"/>

### Section 4: Performance Evaluation for Object Detection
This section contains 3 steps:

1. Compute intersection-over-union between labels and detections (ID_S4_EX1)
2. Compute false-negatives and false-positives (ID_S4_EX2)
3. Compute precision and recall (ID_S4_EX3)

#### Step 1: Compute intersection-over-union between labels and detections (ID_S4_EX1)
The goal of this step is to find pairings between `ground-truth labels` and `detections`, so that we can determine whether an object has been (a) missed `false negative`, (b) successfully detected `true positive` or (c) has been falsely reported `false positive`.

The changes are made in `loop_over_dataset.py`:

```
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
show_only_frames = [50, 51]
model = "darknet"
results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/' + model + '/results_sequence_' + sequence + '_' + model)
sequence = "1"
exec_data = ['pcl_from_rangeimage']
exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance']
exec_tracking = []
exec_visualization = ['show_detection_performance']
configs_det = det.load_configs(model_name="darknet")
```

The function `measure_detection_performance` is updated:

```
####### ID_S4_EX1 START #######     
            #######
            print("student task ID_S4_EX1 ")

            ## step 1 : extract the four corners of the current label bounding-box
            box = label.box
            box_corners_1 = tools.compute_box_corners(box.center_x, box.center_y, box.width, box.length, box.heading)
            ## step 2 : loop over all detected objects
            for obj in detections:

                ## step 3 : extract the four corners of the current detection
                id, x, y, z, h, w, l, yaw = obj
                box_corners_2 = tools.compute_box_corners(x,y,w,l,yaw)
                
                ## step 4 : computer the center distance between label and detection bounding-box in x, y, and z
                dist_x = np.array(box.center_x - x).item()
                dist_y = np.array(box.center_y - y).item()
                dist_z = np.array(box.center_z - z).item()
                
                ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
                poly_box_1 = Polygon(box_corners_1)
                poly_box_2 = Polygon(box_corners_2)
                intersection = poly_box_1.intersection(poly_box_2).area
                union = poly_box_1.union(poly_box_2).area
                iou = intersection / union
                
                ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
                if iou > min_iou:
                    matches_lab_det.append([iou, dist_x, dist_y, dist_z])
                
            #######
            ####### ID_S4_EX1 END ####### 
```

#### Step 2: Compute false-negatives and false-positives (ID_S4_EX2)
The goal of this step is to determine the number of false positives and false negatives for the current frame, based on the pairings between `ground-truth labels` and `detected objects`

The function `measure_detection_performance` is updated:

```
####### ID_S4_EX2 START #######     
    #######
    print("student task ID_S4_EX2")
    
    # compute positives and negatives for precision/recall
    
    ## step 1 : compute the total number of positives present in the scene
    all_positives = labels_valid.sum()

    ## step 2 : compute the number of false negatives
    true_positives = len(ious)
    false_negatives = all_positives - true_positives

    ## step 3 : compute the number of false positives
    false_positives = len(detections) - true_positives
    
    #######
    ####### ID_S4_EX2 END #######
```

#### Step 3: Compute precision and recall (ID_S4_EX3)
After processing all the frames of a sequence, the performance of the object detection algorithm shall now be evaluated.

The changes are made in `loop_over_dataset.py`:

```
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
show_only_frames = [50, 150]
sequence = "1"
exec_data = ['pcl_from_rangeimage']
exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance']
exec_tracking = []
exec_visualization = ['show_detection_performance']
configs_det = det.load_configs(model_name="darknet")
```

The function `compute_performace_stats` is updated:

```
####### ID_S4_EX3 START #######     
    #######    
    print('student task ID_S4_EX3')

    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    positives = sum([item[0] for item in pos_negs])
    true_positives = sum([item[1] for item in pos_negs])
    false_negatives = sum([item[2] for item in pos_negs])
    false_positives = sum([item[3] for item in pos_negs])
    
    ## step 2 : compute precision
    precision = true_positives / (true_positives + false_positives)

    ## step 3 : compute recall 
    recall = true_positives / (true_positives + false_negatives)

    #######    
    ####### ID_S4_EX3 END ####### 
```

The result is: 

`precision: 0.9292604501607717, recall: 0.9444444444444444`

<img src="/img/S4_EX3_1.png"/>

The following changes are made in `loop_over_dataset.py`:

```
configs_det.use_labels_as_objects = True
```

The result is:

`precision: 1.0, recall: 1.0`

<img src="/img/S4_EX3_2.png"/>

## Summary
Within this project, it is clear that LiDAR is effective and should be used for object detection. Evaluating the performance is essential to understand the effectiveness of LiDAR based object detection.

