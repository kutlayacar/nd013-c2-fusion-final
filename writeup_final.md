# Sensor Fusion and Tracking Final

This is the final Project for the second course in the [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213) : Sensor Fusion and Tracking.

In this project, real-world data from [Waymo Open Dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files) and 3D Point Cloud are used for LiDAR based Object Detection.

The Mid-Term project can be found in: [Mid-Term Project](https://github.com/br33x/nd013-c2-fusion-starter-midterm).

## Project Sections

1. Kalman Filter
2. Track Management
3. Data Association
4. Camera-Sensor Fusion

To run this project:
```
python loop_over_dataset.py
```

### Section 1: Kalman Filter
In Section 1 of the final project, an EKF (Extended Kalman Filter) is implemented to track a single real-world target with lidar measurement input over time.

The changes are made in `loop_over_dataset.py`:

```
training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord
show_only_frames = [150, 200]
configs_det = det.load_configs(model_name='fpn_resnet')
configs_det.lim_y = [-5, 10]
exec_detection = []
exec_tracking = ['perform_tracking']
exec_visualization = ['show_tracks']
```

The class `Filter` in `filter.py` is implemented:

```
class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.dim_state = params.dim_state
        self.dt = params.dt
        self.q = params.q

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        F = np.identity(self.dim_state).reshape(self.dim_state, self.dim_state)
        F[0,3] = self.dt
        F[1,4] = self.dt
        F[2,5] = self.dt
    
        return np.matrix(F)
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        Q = np.zeros((self.dim_state, self.dim_state))
        np.fill_diagonal(Q, self.dt * self.q)
        
        return np.matrix(Q)

        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        F = self.F()
        Q = self.Q()
        x = F * track.x
        P = F * track.P * F.T + Q
        track.set_x(x)
        track.set_P(P)

        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        H = meas.sensor.get_H(track.x)
        gamma = self.gamma(track, meas)
        S = self.S(track, meas, H)
        K = track.P * H.T * S.I
        x = track.x + K * gamma
        I = np.identity(self.dim_state)
        P = (I - K * H) * track.P
        track.set_x(x)
        track.set_P(P)
        track.update_attributes(meas)
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############

        return meas.z - meas.sensor.get_hx(track.x)
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        return H * track.P * H.T + meas.R
        
        ############
        # END student code
        ############
```

The result is: 

<img src="/img/F1_1.png"/>
<img src="/img/F1_2.png"/>

### Section 2: Track Management

In this section, the track management is implemented to `initialize` and `delete` tracks, set a `track state` and a `track score`.

The changes are made in `loop_over_dataset.py`:

```
show_only_frames = [65, 100]
configs_det.lim_y = [-5, 15]
```

Following changes are made in `trackmanagement.py`:

The class `Track` is initialized

```
class Track:
    '''Track class with state, covariance, id, score'''

    def __init__(self, meas, id):
        print('creating track no.', id)
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3]  # rotation matrix from sensor to vehicle coordinates

        ############
        # TODO Step 2: initialization:
        # - replace fixed track initialization values by initialization of x and P based on
        # unassigned measurement transformed from sensor to vehicle coordinates
        # - initialize track state and track score with appropriate values
        ############

        coor = np.ones((4, 1))
        coor[0:3] = meas.z[0:3]
        veh = meas.sensor.sens_to_veh * coor

        self.x = np.zeros((6, 1))
        self.x[0:3] = veh[0:3]

        P_position = M_rot * meas.R * M_rot.T

        P_velocity = np.matrix(
            [[params.sigma_p44 ** 2, 0, 0], [0, params.sigma_p55 ** 2, 0], [0, 0, params.sigma_p66 ** 2]])

        self.P = np.zeros((6, 6))
        self.P[0:3, 0:3] = P_position
        self.P[3:6, 3:6] = P_velocity

        self.state = 'initialized'
        self.score = 1. / params.window

        ############
        # END student code
        ############
```

The functions `manage_tracks` and `handle_updated_track` of class `Trackmanagement` are implemented:

```
def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):
        ############
        # TODO Step 2: implement track management:
        # - decrease the track score for unassigned tracks
        # - delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful, but
        # feel free to define your own parameters)
        ############

        # decrease score for unassigned tracks
        for i in unassigned_tracks:
            track = self.track_list[i]
            # check visibility
            if meas_list:  # if not empty
                if meas_list[0].sensor.in_fov(track.x):
                    # your code goes here
                    track.state = 'tentative'
                    if track.score > params.delete_threshold + 1:
                        track.score = params.delete_threshold + 1
                    track.score -= 1. / params.window

                    # delete old tracks
        for track in self.track_list:
            if track.score <= params.delete_threshold:
                if track.P[0, 0] >= params.max_P or track.P[1, 1] >= params.max_P:
                    self.delete_track(track)

        ############
        # END student code
        ############

        # initialize new track with unassigned measurement
        for j in unassigned_meas:
            if meas_list[j].sensor.name == 'lidar':  # only initialize with lidar measurements
                self.init_track(meas_list[j])
```

```
def handle_updated_track(self, track):
        ############
        # TODO Step 2: implement track management for updated tracks:
        # - increase track score
        # - set track state to 'tentative' or 'confirmed'
        ############

        track.score += 1. / params.window
        if track.score > params.confirmed_threshold:
            track.state = 'confirmed'
        else:
            track.state = 'tentative'

        ############
        # END student code
        ############
```

The result is:

<img src="/img/F2.png"/>

### Section 3: Data Association

In Section 3, a `single nearest neighbor data association` is implemented to associate `measurements` to tracks.

The changes are made in `loop_over_dataset.py`:

```
training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
show_only_frames = [0, 200]
configs_det.lim_y = [-25, 25]
```

Following changes are made in `association.py`:

```
def associate(self, track_list, meas_list, KF):

        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############

        # the following only works for at most one track and one measurement
        # self.association_matrix = np.matrix([]) # reset matrix
        self.unassigned_tracks = []  # reset lists
        self.unassigned_meas = []
        association_matrix = []

        for track in track_list:
            r = []
            for meas in meas_list:
                MHD = self.MHD(track, meas, KF)
                sensor = meas.sensor
                if self.gating(MHD, sensor):
                    r.append(MHD)

                else:
                    r.append(np.inf)
            association_matrix.append(r)

        self.unassigned_tracks = np.arange(len(track_list)).tolist()
        self.unassigned_meas = np.arange(len(meas_list)).tolist()

        self.association_matrix = np.matrix(association_matrix)

        ############
        # END student code
        ############
```

```
def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        A = self.association_matrix
        if np.min(A) == np.inf:
            return np.nan, np.nan

        ij_min = np.unravel_index(np.argmin(A, axis=None), A.shape)
        ind_track = ij_min[0]
        ind_meas = ij_min[1]

        A = np.delete(A, ind_track, 0)
        A = np.delete(A, ind_meas, 1)
        self.association_matrix = A

        update_track = self.unassigned_tracks[ind_track]
        update_meas = self.unassigned_meas[ind_meas]

        self.unassigned_tracks.remove(update_track)
        self.unassigned_meas.remove(update_meas)

        ############
        # END student code
        ############
```

```
def gating(self, MHD, sensor):
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############

        return True if MHD < chi2.ppf(params.gating_threshold, df=sensor.dim_meas) else False

        ############
        # END student code
        ############
```

```
def MHD(self, track, meas, KF):
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############

        z = np.matrix(meas.z)
        z_pred = meas.sensor.get_hx(track.x)
        y = z - z_pred
        S = meas.R

        distance = np.sqrt(y.T * S.I * y)

        return distance

        ############
        # END student code
        ############
```

The result is:

<img src="/img/F3.png"/>

### Section 4: Camera-Sensor Fusion

In this section, the `nonlinear camera measurement model` is implemented.

Following changes are made in `measurements.py`:

```
def in_fov(self, x):
        # check if an object x can be seen by this sensor
        ############
        # TODO Step 4: implement a function that returns True if x lies in the sensor's field of view,
        # otherwise False.
        ############

        pos_veh = np.ones((4, 1))
        pos_veh[0:3] = x[0:3]

        pos_sens = self.veh_to_sens * pos_veh

        x, y, z = np.squeeze(pos_sens.A)[:3]

        angle = np.arctan2(y, x)

        if angle >= self.fov[0] and angle <= self.fov[1]:
            return True
        else:
            return False
        ############
        # END student code
        ############
```

```
def get_hx(self, x):
        # calculate nonlinear measurement expectation value h(x)
        if self.name == 'lidar':
            pos_veh = np.ones((4, 1))  # homogeneous coordinates
            pos_veh[0:3] = x[0:3]
            pos_sens = self.veh_to_sens * pos_veh  # transform from vehicle to lidar coordinates
            return pos_sens[0:3]
        elif self.name == 'camera':

            ############
            # TODO Step 4: implement nonlinear camera measurement function h:
            # - transform position estimate from vehicle to camera coordinates
            # - project from camera to image coordinates
            # - make sure to not divide by zero, raise an error if needed
            # - return h(x)
            ############

            pos_veh = np.ones((4, 1))
            pos_veh[0:3] = x[0:3]
            pos_sens = self.veh_to_sens * pos_veh

            x, y, z, = pos_sens[0:3]

            if x <= 0:
                z_pred = np.array([-100, -100])
            else:
                u = self.c_i - self.f_i * y / x
                v = self.c_j - self.f_j * z / x
                z_pred = np.array([u, v])

            return np.matrix(z_pred.reshape(-1, 1))

            ############
            # END student code
            ############
```

```
def generate_measurement(self, num_frame, z, meas_list):
        # generate new measurement from this sensor and add to measurement list
        ############
        # TODO Step 4: remove restriction to lidar in order to include camera as well
        ############

        # if self.name == 'lidar':
        meas = Measurement(num_frame, z, self)
        meas_list.append(meas)
        return meas_list

        ############
        # END student code
        ############
```

```
class Measurement:
    '''Measurement class including measurement values, covariance, timestamp, sensor'''

    def __init__(self, num_frame, z, sensor):
        # create measurement object
        self.t = (num_frame - 1) * params.dt  # time
        if sensor.name == 'lidar':
            sigma_lidar_x = params.sigma_lidar_x  # load params
            sigma_lidar_y = params.sigma_lidar_y
            sigma_lidar_z = params.sigma_lidar_z
            self.z = np.zeros((sensor.dim_meas, 1))  # measurement vector
            self.z[0] = z[0]
            self.z[1] = z[1]
            self.z[2] = z[2]
            self.sensor = sensor  # sensor that generated this measurement
            self.R = np.matrix([[sigma_lidar_x ** 2, 0, 0],  # measurement noise covariance matrix
                                [0, sigma_lidar_y ** 2, 0],
                                [0, 0, sigma_lidar_z ** 2]])

            self.width = z[4]
            self.length = z[5]
            self.height = z[3]
            self.yaw = z[6]
        elif sensor.name == 'camera':

            ############
            # TODO Step 4: initialize camera measurement including z, R, and sensor
            ############

            self.z = np.zeros((sensor.dim_meas, 1))
            self.z[0][0] = z[0]
            self.z[1][0] = z[1]
            self.sensor = sensor
            sig_cam_i = params.sigma_cam_i
            sig_cam_j = params.sigma_cam_j
            self.R = np.matrix([[sig_cam_i ** 2, 0], [0, sig_cam_j ** 2]])

            ############
            # END student code
            ############
```

The result is:

<img src="/img/F4.png"/>

## Summary

Within this project, it is clear that `Sensor Fusion` is important for a stabilized tracking. While cameras offer textured and color/brightness/contrast based images, LiDAR is extremely beneficial for detection in low quality conditions such as rainy weather or dark/blurry images. In conclusion, Sensor Fusion is crucial to track objects precisely in different conditions.


